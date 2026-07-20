"""Read and write runtime summaries, diagnostics, restart state, and tables."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

from jax.typing import ArrayLike
import numpy as np

from gkx.artifacts.spectral_layout import (
    _condense_kx,
    _condense_ky,
    _condense_kykx,
    _require_netcdf4,
)
from gkx.diagnostics import (
    ResolvedDiagnostics,
    SimulationDiagnostics,
    total_energy,
)
from gkx.workflows.runtime.diagnostic_arrays import (
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



def _nonlinear_summary(result: Any) -> dict[str, Any]:
    diag = result.diagnostics
    payload: dict[str, Any] = {
        "kind": "nonlinear",
        "ky_selected": None
        if result.ky_selected is None
        else float(result.ky_selected),
        "kx_selected": None
        if result.kx_selected is None
        else float(result.kx_selected),
        "n_state_shape": None
        if result.state is None
        else list(np.asarray(result.state).shape),
    }
    if diag is not None:
        payload.update(
            {
                "n_samples": int(np.asarray(diag.t).size),
                "t_last": float(np.asarray(diag.t)[-1])
                if np.asarray(diag.t).size
                else 0.0,
                "dt_mean": float(np.asarray(diag.dt_mean)),
                "gamma_last": float(np.asarray(diag.gamma_t)[-1])
                if np.asarray(diag.gamma_t).size
                else 0.0,
                "omega_last": float(np.asarray(diag.omega_t)[-1])
                if np.asarray(diag.omega_t).size
                else 0.0,
                "Wg_last": float(np.asarray(diag.Wg_t)[-1])
                if np.asarray(diag.Wg_t).size
                else 0.0,
                "Wphi_last": float(np.asarray(diag.Wphi_t)[-1])
                if np.asarray(diag.Wphi_t).size
                else 0.0,
                "Wapar_last": float(np.asarray(diag.Wapar_t)[-1])
                if np.asarray(diag.Wapar_t).size
                else 0.0,
                "heat_flux_last": (
                    float(np.asarray(diag.heat_flux_t)[-1])
                    if np.asarray(diag.heat_flux_t).size
                    else 0.0
                ),
                "particle_flux_last": (
                    float(np.asarray(diag.particle_flux_t)[-1])
                    if np.asarray(diag.particle_flux_t).size
                    else 0.0
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


def write_runtime_nonlinear_table_artifacts(
    out: str | Path, result: Any
) -> dict[str, str]:
    """Write non-NetCDF nonlinear summary/diagnostics/state artifacts."""

    out_path = Path(out)
    base = _artifact_base(out_path)
    summary_path = (
        out_path if out_path.suffix.lower() == ".json" else Path(f"{base}.summary.json")
    )
    csv_path = (
        out_path
        if out_path.suffix.lower() == ".csv"
        else Path(f"{base}.diagnostics.csv")
    )

    _write_json(summary_path, _nonlinear_summary(result))
    paths = {"summary": str(summary_path)}
    diag: SimulationDiagnostics | None = result.diagnostics
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
        if diag.turbulent_heating_t is not None:
            cols.append(_flatten_series(np.asarray(diag.turbulent_heating_t)))
            headers.append("turbulent_heating")
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
        if diag.turbulent_heating_species_t is not None:
            turb_heat_s = np.asarray(diag.turbulent_heating_species_t)
            if turb_heat_s.ndim == 1:
                turb_heat_s = turb_heat_s[:, None]
            for i in range(turb_heat_s.shape[1]):
                cols.append(turb_heat_s[:, i])
                headers.append(f"turbulent_heating_s{i}")
        _write_csv(csv_path, headers=headers, cols=cols)
        paths["diagnostics"] = str(csv_path)

    state_path = _write_state(
        base, None if result.state is None else np.asarray(result.state)
    )
    if state_path is not None:
        paths["state"] = str(state_path)
    return paths



def _dealiased_kx_count(nx_full: int) -> int:
    return 1 + 2 * ((int(nx_full) - 1) // 3)


def _dealiased_ky_count(ny_full: int) -> int:
    return 1 + ((int(ny_full) - 1) // 3)


def _dealiased_kx_indices(nx_full: int) -> np.ndarray:
    nx = int(nx_full)
    split = 1 + ((nx - 1) // 3)
    if nx <= 1:
        return np.array([0], dtype=np.int32)
    neg = np.arange(2 * nx // 3 + 1, nx, dtype=np.int32)
    pos = np.arange(0, split, dtype=np.int32)
    return np.concatenate([neg, pos], axis=0)


def _expand_positive_ky_to_full(
    state_positive_ky: np.ndarray, *, ny_full: int
) -> np.ndarray:
    state = np.asarray(state_positive_ky)
    if state.ndim != 6:
        raise ValueError("state_positive_ky must have shape (Ns, Nl, Nm, Nyc, Nx, Nz)")
    nyc = state.shape[3]
    expected_nyc = int(ny_full) // 2 + 1
    if nyc != expected_nyc:
        raise ValueError(
            f"positive-ky state Nyc={nyc} does not match ny_full={ny_full}"
        )
    neg_hi = nyc - 1 if (int(ny_full) % 2) == 0 else nyc
    neg = np.conj(state[..., 1:neg_hi, :, :])[..., ::-1, :, :]
    nx = state.shape[4]
    if nx > 1:
        kx_neg = np.concatenate(([0], np.arange(nx - 1, 0, -1)))
        neg = neg[..., kx_neg, :]
    return np.concatenate([state, neg], axis=3)


def _expand_netcdf_restart_state_to_full_positive_ky(
    state_active: np.ndarray,
    *,
    ny_full: int,
    nx_full: int,
) -> np.ndarray:
    state = np.asarray(state_active)
    if state.ndim != 6:
        raise ValueError("state_active must have shape (Ns, Nl, Nm, Naky, Nakx, Nz)")
    nspec, nl, nm, naky, nakx, nz = state.shape
    nyc_full = int(ny_full) // 2 + 1
    expected_naky = _dealiased_ky_count(int(ny_full))
    expected_nakx = _dealiased_kx_count(int(nx_full))
    if naky != expected_naky:
        raise ValueError(
            f"restart Nky={naky} does not match ny_full={ny_full} (expected {expected_naky})"
        )
    if nakx != expected_nakx:
        raise ValueError(
            f"restart Nkx={nakx} does not match nx_full={nx_full} (expected {expected_nakx})"
        )
    out = np.zeros((nspec, nl, nm, nyc_full, int(nx_full), nz), dtype=np.complex64)
    out[..., :naky, _dealiased_kx_indices(int(nx_full)), :] = state
    return out


def _expand_netcdf_restart_state_full_ky(
    state_active: np.ndarray,
    *,
    nx_full: int,
) -> np.ndarray:
    """Expand a NetCDF restart that already stores the full ``ky`` axis."""

    state = np.asarray(state_active)
    if state.ndim != 6:
        raise ValueError("state_active must have shape (Ns, Nl, Nm, Ny, Nakx, Nz)")
    nspec, nl, nm, ny_full, nakx, nz = state.shape
    expected_nakx = _dealiased_kx_count(int(nx_full))
    if nakx != expected_nakx:
        raise ValueError(
            f"restart Nkx={nakx} does not match nx_full={nx_full} (expected {expected_nakx})"
        )
    out = np.zeros((nspec, nl, nm, int(ny_full), int(nx_full), nz), dtype=np.complex64)
    out[..., _dealiased_kx_indices(int(nx_full)), :] = state
    return out


def write_netcdf_restart_state(path: str | Path, state: ArrayLike) -> Path:
    """Write a restart state in flat complex64 restart layout."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(state, dtype=np.complex64).tofile(out)
    return out


def load_netcdf_restart_state(
    path: str | Path,
    *,
    nspecies: int,
    Nl: int,
    Nm: int,
    ny: int,
    nx: int,
    nz: int,
) -> np.ndarray:
    """Load a NetCDF restart file into GKX's full Hermitian layout."""

    try:
        from netCDF4 import Dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError("netCDF4 is required to load NetCDF restart files") from exc

    with Dataset(Path(path), "r") as root:
        if "G" not in root.variables:
            raise ValueError(f"restart file {path} does not contain variable 'G'")
        raw = np.asarray(root.variables["G"][:], dtype=float)
    if raw.ndim != 7 or raw.shape[-1] != 2:
        raise ValueError(f"unexpected NetCDF restart G shape {raw.shape}")
    state_active = raw[..., 0] + 1j * raw[..., 1]
    state_active = np.asarray(
        np.transpose(state_active, (0, 2, 1, 5, 4, 3)), dtype=np.complex64
    )
    if state_active.shape[:3] != (int(nspecies), int(Nl), int(Nm)):
        raise ValueError(
            f"restart state shape {state_active.shape[:3]} does not match requested {(int(nspecies), int(Nl), int(Nm))}"
        )
    if state_active.shape[-1] != int(nz):
        raise ValueError(
            f"restart Nz={state_active.shape[-1]} does not match requested {int(nz)}"
        )
    if state_active.shape[3] == int(ny):
        return _expand_netcdf_restart_state_full_ky(state_active, nx_full=nx)
    positive_ky = _expand_netcdf_restart_state_to_full_positive_ky(
        state_active, ny_full=ny, nx_full=nx
    )
    return _expand_positive_ky_to_full(positive_ky, ny_full=ny)

def _resolved_species_time(arr: Any | None, *, fallback: np.ndarray) -> np.ndarray:
    if arr is None:
        return np.asarray(fallback, dtype=np.float32)
    return np.sum(np.asarray(arr, dtype=np.float32), axis=-1)


def _read_optional_var(group: Any, name: str) -> np.ndarray | None:
    if name not in group.variables:
        return None
    var = group.variables[name]
    arr = np.asarray(var[:])
    dims = tuple(getattr(var, "dimensions", ()))
    if dims and dims[-1] == "ri":
        return np.asarray(arr[..., 0] + 1j * arr[..., 1])
    return arr


def _resolve_restart_path(out: str | Path, cfg: Any, *, for_write: bool) -> Path:
    configured = (
        cfg.output.restart_to_file if for_write else cfg.output.restart_from_file
    )
    if configured is not None:
        return Path(configured)
    base = _netcdf_bundle_base(Path(out))
    return Path(f"{base}.restart.nc")


def _condense_resolved_for_output(
    resolved: ResolvedDiagnostics | None,
) -> ResolvedDiagnostics | None:
    if resolved is None:
        return None
    payload: dict[str, np.ndarray | None] = {}
    for field in ResolvedDiagnostics.__dataclass_fields__.values():
        value = getattr(resolved, field.name)
        if value is None:
            payload[field.name] = None
        elif field.name.endswith(("_kxt", "_kxst")):
            payload[field.name] = _condense_kx(np.asarray(value))
        elif field.name.endswith(("_kyt", "_kyst")):
            payload[field.name] = _condense_ky(np.asarray(value))
        elif field.name.endswith(("_kxkyt", "_kxkyst")):
            payload[field.name] = _condense_kykx(np.asarray(value))
        else:
            payload[field.name] = np.asarray(value)
    return ResolvedDiagnostics(**payload)


def _condense_diagnostics_for_netcdf_output(
    diag: SimulationDiagnostics,
) -> SimulationDiagnostics:
    # Nonlinear NetCDF output artifacts do not persist the monitored complex mode trace.
    # Drop it when appending from an existing artifact so restart concatenation
    # preserves the exact on-disk schema instead of mixing persisted and transient
    # diagnostics.
    return replace(
        diag, phi_mode_t=None, resolved=_condense_resolved_for_output(diag.resolved)
    )


def load_nonlinear_netcdf_diagnostics(path: str | Path) -> SimulationDiagnostics:
    Dataset = _require_netcdf4()
    with Dataset(Path(path), "r") as root:
        grids = root.groups["Grids"]
        diag_group = root.groups["Diagnostics"]
        time_vals = np.asarray(grids.variables["time"][:], dtype=np.float64)
        wg_st = np.asarray(diag_group.variables["Wg_st"][:], dtype=np.float32)
        wphi_st = np.asarray(diag_group.variables["Wphi_st"][:], dtype=np.float32)
        wapar_st = np.asarray(diag_group.variables["Wapar_st"][:], dtype=np.float32)
        heat_st = np.asarray(diag_group.variables["HeatFlux_st"][:], dtype=np.float32)
        pflux_st = np.asarray(
            diag_group.variables["ParticleFlux_st"][:], dtype=np.float32
        )
        turb_heat_st = _read_optional_var(diag_group, "TurbulentHeating_st")
        resolved_payload = {
            field.name: _read_optional_var(diag_group, field.name)
            for field in ResolvedDiagnostics.__dataclass_fields__.values()
        }
    if turb_heat_st is None:
        turb_heat_st = np.zeros_like(heat_st)
    dt_t = (
        np.diff(np.concatenate(([0.0], time_vals)))
        if time_vals.size
        else np.asarray([], dtype=np.float64)
    )
    dt_mean = np.asarray(
        np.mean(dt_t[dt_t > 0.0]) if np.any(dt_t > 0.0) else 0.0, dtype=np.float64
    )
    Wg_t = np.sum(wg_st, axis=1)
    Wphi_t = np.sum(wphi_st, axis=1)
    Wapar_t = np.sum(wapar_st, axis=1)
    heat_t = np.sum(heat_st, axis=1)
    pflux_t = np.sum(pflux_st, axis=1)
    turb_heat_t = np.sum(np.asarray(turb_heat_st, dtype=np.float32), axis=1)
    return SimulationDiagnostics(
        t=time_vals,
        dt_t=dt_t,
        dt_mean=dt_mean,
        gamma_t=np.zeros_like(time_vals, dtype=np.float32),
        omega_t=np.zeros_like(time_vals, dtype=np.float32),
        Wg_t=Wg_t,
        Wphi_t=Wphi_t,
        Wapar_t=Wapar_t,
        heat_flux_t=heat_t,
        particle_flux_t=pflux_t,
        energy_t=np.asarray(total_energy(Wg_t, Wphi_t, Wapar_t), dtype=np.float32),
        heat_flux_species_t=heat_st,
        particle_flux_species_t=pflux_st,
        turbulent_heating_t=turb_heat_t,
        turbulent_heating_species_t=np.asarray(turb_heat_st, dtype=np.float32),
        phi_mode_t=None,
        resolved=ResolvedDiagnostics(**resolved_payload),
    )




@dataclass(frozen=True)
class DiagnosticTimeSeries:
    """Single benchmark-facing time series loaded from an ``out.nc`` artifact."""

    t: np.ndarray
    values: np.ndarray
    variable: str
    source_path: str


def _decode_netcdf_values(var) -> np.ndarray:
    raw = np.asarray(var[:])
    dims = tuple(getattr(var, "dimensions", ()))
    if dims and dims[-1] == "ri":
        return np.asarray(raw[..., 0] + 1j * raw[..., 1], dtype=np.complex128)
    return raw


def _extract_diagnostic_values(
    values: np.ndarray,
    *,
    variable: str,
    kx_index: int | None,
) -> np.ndarray:
    if values.ndim == 1:
        return values
    if values.ndim == 2:
        if kx_index is None:
            raise ValueError(
                f"diagnostics variable {variable!r} requires kx_index for 2D extraction"
            )
        return values[:, int(kx_index)]
    raise ValueError(
        f"diagnostics variable {variable!r} must reduce to a 1D time series"
    )


def _load_netcdf_time_axis(
    ds,
    *,
    src: Path,
    time_group: str,
    time_var: str,
) -> np.ndarray:
    if time_group in ds.groups and time_var in ds.groups[time_group].variables:
        return np.asarray(ds.groups[time_group].variables[time_var][:], dtype=float)
    if time_var in ds.variables:
        return np.asarray(ds.variables[time_var][:], dtype=float)
    raise ValueError(f"missing time variable {time_group}/{time_var} in {src}")


def _load_diagnostic_variable(
    ds,
    *,
    src: Path,
    diagnostics_group: str,
    variable: str,
    kx_index: int | None,
) -> np.ndarray:
    diag_group = ds.groups.get(diagnostics_group)
    if diag_group is None:
        raise ValueError(f"missing NetCDF group {diagnostics_group!r} in {src}")
    if variable not in diag_group.variables:
        raise ValueError(f"missing diagnostics variable {variable!r} in {src}")
    raw = _decode_netcdf_values(diag_group.variables[variable])
    return _extract_diagnostic_values(raw, variable=variable, kx_index=kx_index)


def _align_complex_phase(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    nz = finite & (np.abs(values) > 1.0e-30)
    if np.any(nz):
        first = values[np.flatnonzero(nz)[0]]
        return values * np.exp(-1j * np.angle(first))
    return values


def _select_complex_component(
    values: np.ndarray,
    *,
    component: str,
    align_phase: bool,
) -> np.ndarray:
    values_arr = _align_complex_phase(values) if align_phase else values
    component_key = str(component).lower()
    if component_key == "complex":
        return values_arr
    if component_key == "real":
        return np.real(values_arr)
    if component_key == "imag":
        return np.imag(values_arr)
    if component_key == "abs":
        return np.abs(values_arr)
    raise ValueError("component must be one of {'real', 'imag', 'abs', 'complex'}")


def _select_real_component(values: np.ndarray, *, component: str) -> np.ndarray:
    if component not in {"real", "abs"}:
        raise ValueError("real diagnostics only support component='real' or 'abs'")
    if component == "abs":
        return np.abs(values)
    return np.asarray(values, dtype=float)


def _select_series_component(
    values: np.ndarray,
    *,
    component: str,
    align_phase: bool,
) -> np.ndarray:
    values_arr = np.asarray(values)
    if np.iscomplexobj(values_arr):
        return _select_complex_component(
            values_arr,
            component=component,
            align_phase=align_phase,
        )
    return _select_real_component(values_arr, component=component)


def load_diagnostic_time_series(
    path: str | Path,
    *,
    variable: str,
    diagnostics_group: str = "Diagnostics",
    time_group: str = "Grids",
    time_var: str = "time",
    kx_index: int | None = None,
    component: str = "real",
    align_phase: bool = False,
) -> DiagnosticTimeSeries:
    """Load a 1D diagnostics time series from a grouped NetCDF output artifact."""

    src = Path(path)
    import netCDF4 as nc

    with nc.Dataset(src) as ds:
        values = _load_diagnostic_variable(
            ds,
            src=src,
            diagnostics_group=diagnostics_group,
            variable=variable,
            kx_index=kx_index,
        )
        t = _load_netcdf_time_axis(
            ds,
            src=src,
            time_group=time_group,
            time_var=time_var,
        )

    if t.ndim != 1 or t.size != values.size:
        raise ValueError(
            f"time axis for {variable!r} must be one-dimensional and match the diagnostics length"
        )
    selected = _select_series_component(
        values,
        component=component,
        align_phase=align_phase,
    )

    return DiagnosticTimeSeries(
        t=t,
        values=np.asarray(selected),
        variable=str(variable),
        source_path=str(src),
    )



__all__ = [
    "DiagnosticTimeSeries",
    "_artifact_base",
    "_condense_diagnostics_for_netcdf_output",
    "_ensure_parent",
    "_flatten_series",
    "_is_netcdf_output_target",
    "_netcdf_bundle_base",
    "_resolve_restart_path",
    "_resolved_species_time",
    "_write_csv",
    "_write_json",
    "_write_state",
    "load_diagnostic_time_series",
    "load_netcdf_restart_state",
    "load_nonlinear_netcdf_diagnostics",
    "validate_finite_array",
    "validate_finite_runtime_result",
    "write_netcdf_restart_state",
    "write_quasilinear_artifacts",
    "write_runtime_linear_artifacts",
    "write_runtime_linear_scan_artifacts",
    "write_runtime_nonlinear_table_artifacts",
]
