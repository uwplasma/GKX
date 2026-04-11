"""Structured runtime artifact writers for CLI and benchmark tooling."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np

from spectraxgk.geometry import ensure_flux_tube_geometry_data
from spectraxgk.grids import build_spectral_grid, real_fft_ordered_kx, real_fft_unique_ky
from spectraxgk.diagnostics import SimulationDiagnostics, ResolvedDiagnostics, total_energy
from spectraxgk.linear import build_linear_cache
from spectraxgk.runtime import (
    RuntimeNonlinearResult,
    _concat_gx_diagnostics,
    build_runtime_geometry,
    build_runtime_linear_params,
    run_runtime_nonlinear,
)


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


def _resolved_species_time(arr: Any | None, *, fallback: np.ndarray) -> np.ndarray:
    if arr is None:
        return np.asarray(fallback, dtype=np.float32)
    return np.sum(np.asarray(arr, dtype=np.float32), axis=-1)


def _read_optional_var(group: Any, name: str) -> np.ndarray | None:
    if name not in group.variables:
        return None
    return np.asarray(group.variables[name][:])


def _resolve_restart_path(out: str | Path, cfg: Any, *, for_write: bool) -> Path:
    configured = cfg.output.restart_to_file if for_write else cfg.output.restart_from_file
    if configured is not None:
        return Path(configured)
    base = _gx_bundle_base(Path(out))
    return Path(f"{base}.restart.nc")


def _condense_resolved_for_output(resolved: ResolvedDiagnostics | None) -> ResolvedDiagnostics | None:
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


def _condense_gx_diagnostics_for_output(diag: SimulationDiagnostics) -> SimulationDiagnostics:
    return replace(diag, resolved=_condense_resolved_for_output(diag.resolved))


def load_runtime_nonlinear_gx_diagnostics(path: str | Path) -> SimulationDiagnostics:
    Dataset = _require_netcdf4()
    with Dataset(Path(path), "r") as root:
        grids = root.groups["Grids"]
        diag_group = root.groups["Diagnostics"]
        time_vals = np.asarray(grids.variables["time"][:], dtype=np.float64)
        wg_st = np.asarray(diag_group.variables["Wg_st"][:], dtype=np.float32)
        wphi_st = np.asarray(diag_group.variables["Wphi_st"][:], dtype=np.float32)
        wapar_st = np.asarray(diag_group.variables["Wapar_st"][:], dtype=np.float32)
        heat_st = np.asarray(diag_group.variables["HeatFlux_st"][:], dtype=np.float32)
        pflux_st = np.asarray(diag_group.variables["ParticleFlux_st"][:], dtype=np.float32)
        turb_heat_st = _read_optional_var(diag_group, "TurbulentHeating_st")
        resolved_payload = {
            field.name: _read_optional_var(diag_group, field.name)
            for field in ResolvedDiagnostics.__dataclass_fields__.values()
        }
    if turb_heat_st is None:
        turb_heat_st = np.zeros_like(heat_st)
    dt_t = np.diff(np.concatenate(([0.0], time_vals))) if time_vals.size else np.asarray([], dtype=np.float64)
    dt_mean = np.asarray(np.mean(dt_t[dt_t > 0.0]) if np.any(dt_t > 0.0) else 0.0, dtype=np.float64)
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


def run_runtime_nonlinear_with_artifacts(
    cfg: Any,
    *,
    out: str | Path | None,
    ky_target: float,
    kx_target: float | None = None,
    Nl: int | None = None,
    Nm: int | None = None,
    dt: float | None = None,
    steps: int | None = None,
    method: str | None = None,
    sample_stride: int | None = None,
    diagnostics_stride: int | None = None,
    laguerre_mode: str | None = None,
    diagnostics: bool | None = None,
    show_progress: bool = False,
    status_callback: Any = None,
) -> tuple[RuntimeNonlinearResult, dict[str, str]]:
    out_path = None if out is None else Path(out)
    gx_target = out_path is not None and _is_gx_netcdf_target(out_path)
    diagnostics_on = bool(cfg.time.diagnostics if diagnostics is None else diagnostics)
    if gx_target and not diagnostics_on:
        raise ValueError("GX-style nonlinear NetCDF artifacts require diagnostics output")

    cfg_run = cfg
    restart_from: Path | None = None
    restart_to: Path | None = None
    if gx_target:
        assert out_path is not None
        restart_from = _resolve_restart_path(out_path, cfg, for_write=False)
        restart_to = _resolve_restart_path(out_path, cfg, for_write=True)
    resume_requested = bool(getattr(cfg.output, "restart", False)) or cfg.init.init_file is not None
    if gx_target and cfg.init.init_file is None:
        if bool(getattr(cfg.output, "restart_if_exists", False)) and restart_from is not None and restart_from.exists():
            resume_requested = True
            cfg_run = replace(
                cfg_run,
                init=replace(
                    cfg_run.init,
                    init_file=str(restart_from),
                    init_file_scale=float(getattr(cfg.output, "restart_scale", 1.0)),
                    init_file_mode="add" if bool(getattr(cfg.output, "restart_with_perturb", False)) else "replace",
                ),
            )
        elif bool(getattr(cfg.output, "restart", False)) and restart_from is not None:
            if not restart_from.exists():
                raise FileNotFoundError(f"restart file not found: {restart_from}")
            cfg_run = replace(
                cfg_run,
                init=replace(
                    cfg_run.init,
                    init_file=str(restart_from),
                    init_file_scale=float(getattr(cfg.output, "restart_scale", 1.0)),
                    init_file_mode="add" if bool(getattr(cfg.output, "restart_with_perturb", False)) else "replace",
                ),
            )
    elif cfg.init.init_file is not None and bool(getattr(cfg.output, "restart_with_perturb", False)):
        cfg_run = replace(
            cfg_run,
            init=replace(
                cfg_run.init,
                init_file_scale=float(getattr(cfg.output, "restart_scale", 1.0)),
                init_file_mode="add",
            ),
        )

    cumulative_diag: SimulationDiagnostics | None = None
    history_from_file = False
    if gx_target and resume_requested and bool(getattr(cfg.output, "append_on_restart", True)):
        assert out_path is not None
        history_path = Path(f"{_gx_bundle_base(out_path)}.out.nc")
        if history_path.exists():
            cumulative_diag = load_runtime_nonlinear_gx_diagnostics(history_path)
            history_from_file = True

    checkpoint_steps: int | None = None
    if gx_target and bool(getattr(cfg.output, "save_for_restart", True)):
        if getattr(cfg.time, "nstep_restart", None) is not None and int(cfg.time.nstep_restart) > 0:
            checkpoint_steps = int(cfg.time.nstep_restart)
        elif int(getattr(cfg.output, "nsave", 0)) > 0:
            checkpoint_steps = int(cfg.output.nsave)

    if steps is not None:
        remaining_steps: int | None = int(steps)
    elif bool(cfg.time.fixed_dt):
        remaining_steps = int(round(float(cfg.time.t_max) / float(cfg.time.dt if dt is None else dt)))
    else:
        remaining_steps = None

    time_offset = 0.0
    if cumulative_diag is not None and np.asarray(cumulative_diag.t).size:
        time_offset = float(np.asarray(cumulative_diag.t)[-1])

    result_final: RuntimeNonlinearResult | None = None
    paths: dict[str, str] = {}
    while True:
        chunk_steps = remaining_steps
        if checkpoint_steps is not None:
            chunk_steps = checkpoint_steps if remaining_steps is None else min(int(remaining_steps), checkpoint_steps)
        result_chunk = run_runtime_nonlinear(
            cfg_run,
            ky_target=ky_target,
            kx_target=kx_target,
            Nl=Nl,
            Nm=Nm,
            dt=dt,
            steps=chunk_steps,
            method=method,
            sample_stride=sample_stride,
            diagnostics_stride=diagnostics_stride,
            laguerre_mode=laguerre_mode,
            diagnostics=diagnostics,
            return_state=gx_target,
            show_progress=show_progress,
            status_callback=status_callback,
        )
        result_effective = result_chunk
        if result_chunk.diagnostics is not None:
            diag_chunk = result_chunk.diagnostics
            if history_from_file:
                diag_chunk = _condense_gx_diagnostics_for_output(diag_chunk)
            if time_offset != 0.0:
                diag_chunk = replace(diag_chunk, t=np.asarray(diag_chunk.t) + time_offset)
            cumulative_diag = diag_chunk if cumulative_diag is None else _concat_gx_diagnostics([cumulative_diag, diag_chunk])
            time_offset = float(np.asarray(cumulative_diag.t)[-1]) if np.asarray(cumulative_diag.t).size else time_offset
            result_effective = replace(result_chunk, diagnostics=cumulative_diag, t=np.asarray(cumulative_diag.t))
        result_final = result_effective

        if out_path is not None:
            paths = write_runtime_nonlinear_artifacts(out_path, result_effective, cfg)

        if checkpoint_steps is None:
            break
        if remaining_steps is not None:
            assert chunk_steps is not None
            remaining_steps -= int(chunk_steps)
            if remaining_steps <= 0:
                break
        elif result_effective.diagnostics is None or time_offset >= float(cfg.time.t_max) - 1.0e-12:
            break
        if restart_to is None:
            break
        cfg_run = replace(
            cfg,
            init=replace(
                cfg.init,
                init_file=str(restart_to),
                init_file_scale=1.0,
                init_file_mode="replace",
            ),
        )

    if result_final is None:
        raise RuntimeError("nonlinear runtime produced no result")
    return result_final, paths


def _require_netcdf4():
    try:
        from netCDF4 import Dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError("netCDF4 is required to write GX-style NetCDF runtime artifacts") from exc
    return Dataset


def _real_space_axis(length: int, extent: float) -> np.ndarray:
    return np.linspace(0.0, float(extent), int(length), endpoint=False, dtype=np.float32)


def _gx_active_kx_count(nx: int) -> int:
    return 1 + 2 * ((int(nx) - 1) // 3)


def _gx_active_ky_count(ny: int) -> int:
    return 1 + ((int(ny) - 1) // 3)


def _gx_active_kx_indices(nx: int) -> np.ndarray:
    nx_use = int(nx)
    split = 1 + ((nx_use - 1) // 3)
    if nx_use <= 1:
        return np.array([0], dtype=np.int32)
    neg = np.arange(2 * nx_use // 3 + 1, nx_use, dtype=np.int32)
    pos = np.arange(0, split, dtype=np.int32)
    return np.concatenate([neg, pos], axis=0)


def _gx_active_ky_indices(ny: int) -> np.ndarray:
    return np.arange(_gx_active_ky_count(int(ny)), dtype=np.int32)


def _gx_active_kx_values(kx: np.ndarray) -> np.ndarray:
    kx_arr = np.asarray(kx, dtype=np.float32)
    return kx_arr[_gx_active_kx_indices(kx_arr.shape[0])]


def _gx_active_ky_values(ky: np.ndarray) -> np.ndarray:
    ky_arr = np.asarray(ky, dtype=np.float32)
    nyc = 1 + ky_arr.shape[0] // 2
    return np.abs(ky_arr[:nyc])[: _gx_active_ky_count(int(ky_arr.shape[0]))]


def _take_axis(arr: np.ndarray, indices: np.ndarray, axis: int) -> np.ndarray:
    return np.take(np.asarray(arr), indices.astype(np.int32, copy=False), axis=axis)


def _spectral_to_ri(field: np.ndarray) -> np.ndarray:
    field_arr = np.asarray(field)
    if field_arr.ndim != 3:
        raise ValueError("field must have shape (Ny, Nx, Nz)")
    return np.stack([np.real(field_arr), np.imag(field_arr)], axis=-1).astype(np.float32, copy=False)


def _spectral_to_xy(field: np.ndarray) -> np.ndarray:
    xy = np.fft.ifft2(np.asarray(field), axes=(0, 1))
    return np.real(xy).astype(np.float32, copy=False)


def _restart_to_gx_layout(state: np.ndarray) -> np.ndarray:
    state_arr = np.asarray(state)
    if state_arr.ndim == 5:
        state_arr = state_arr[None, ...]
    if state_arr.ndim != 6:
        raise ValueError("nonlinear state must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)")
    ky_idx = _gx_active_ky_indices(state_arr.shape[3])
    kx_idx = _gx_active_kx_indices(state_arr.shape[4])
    state_arr = _take_axis(state_arr, ky_idx, axis=3)
    state_arr = _take_axis(state_arr, kx_idx, axis=4)
    gx = np.transpose(state_arr, (0, 2, 1, 5, 4, 3))
    return np.stack([np.real(gx), np.imag(gx)], axis=-1).astype(np.float32, copy=False)


def _species_matrix(total: np.ndarray, nspecies: int, species_values: np.ndarray | None) -> np.ndarray:
    total_arr = np.asarray(total, dtype=np.float32)
    ns = max(int(nspecies), 1)
    if species_values is not None:
        arr = np.asarray(species_values, dtype=np.float32)
        if arr.ndim == 1:
            return arr[:, None]
        return arr
    return np.broadcast_to((total_arr / float(ns))[:, None], (total_arr.shape[0], ns)).copy()


def _maybe_var(group: Any, name: str, dtype: str, dims: tuple[str, ...], values: np.ndarray) -> None:
    var = group.createVariable(name, dtype, dims)
    var[...] = values


def _write_runtime_root_metadata(root: Any, cfg: Any, *, nspecies: int, nl: int, nm: int) -> None:
    root.createVariable("ny", "i4", ())[:] = np.int32(cfg.grid.Ny)
    root.createVariable("nx", "i4", ())[:] = np.int32(cfg.grid.Nx)
    root.createVariable("ntheta", "i4", ())[:] = np.int32(cfg.grid.ntheta if cfg.grid.ntheta is not None else cfg.grid.Nz)
    root.createVariable("nhermite", "i4", ())[:] = np.int32(nm)
    root.createVariable("nlaguerre", "i4", ())[:] = np.int32(nl)
    root.createVariable("nspecies", "i4", ())[:] = np.int32(nspecies)
    root.createVariable("nperiod", "i4", ())[:] = np.int32(cfg.grid.nperiod if cfg.grid.nperiod is not None else 1)
    root.createVariable("debug", "i4", ())[:] = np.int32(0)
    code_info = root.createVariable("code_info", "i4", ())
    code_info[:] = np.int32(1)
    code_info.setncattr("value", "spectrax-gk")


def _gx_active_field(field: np.ndarray, *, ky_axis: int = 0, kx_axis: int = 1) -> np.ndarray:
    field_arr = np.asarray(field)
    ky_idx = _gx_active_ky_indices(field_arr.shape[ky_axis])
    kx_idx = _gx_active_kx_indices(field_arr.shape[kx_axis])
    return _take_axis(_take_axis(field_arr, ky_idx, axis=ky_axis), kx_idx, axis=kx_axis)


def _spectral_species_to_ri(field: np.ndarray) -> np.ndarray:
    field_arr = np.asarray(field)
    if field_arr.ndim != 4:
        raise ValueError("field must have shape (Ns, Ny, Nx, Nz)")
    return np.stack([np.real(field_arr), np.imag(field_arr)], axis=-1).astype(np.float32, copy=False)


def _state_basis_moments(state: np.ndarray) -> dict[str, np.ndarray]:
    state_arr = np.asarray(state)
    if state_arr.ndim != 6:
        raise ValueError("state must have shape (Ns, Nl, Nm, Ny, Nx, Nz)")
    ns, nl, nm, _ny, _nx, nz = state_arr.shape
    zeros = np.zeros((ns, state_arr.shape[3], state_arr.shape[4], nz), dtype=state_arr.dtype)
    density = state_arr[:, 0, 0, ...] if nl >= 1 and nm >= 1 else zeros
    upar = state_arr[:, 0, 1, ...] if nl >= 1 and nm >= 2 else zeros
    tpar = np.sqrt(2.0, dtype=np.float32) * state_arr[:, 0, 2, ...] if nl >= 1 and nm >= 3 else zeros
    tperp = state_arr[:, 1, 0, ...] if nl >= 2 and nm >= 1 else zeros
    return {
        "Density": density,
        "Upar": upar,
        "Tpar": tpar,
        "Tperp": tperp,
    }


def _particle_moments(state: np.ndarray, cfg: Any) -> dict[str, np.ndarray]:
    state_arr = np.asarray(state)
    ns, nl, nm, _ny, _nx, _nz = state_arr.shape
    grid = build_spectral_grid(cfg.grid)
    geom = build_runtime_geometry(cfg)
    params = build_runtime_linear_params(cfg, Nm=nm, geom=geom)
    cache = build_linear_cache(grid, geom, params, nl, nm)
    Jl = np.asarray(cache.Jl)
    JlB = np.asarray(cache.JlB)
    if Jl.ndim == 4:
        Jl = Jl[None, ...]
    if JlB.ndim == 4:
        JlB = JlB[None, ...]
    sqrt_b = np.sqrt(np.maximum(np.asarray(cache.kperp2, dtype=np.float32), 0.0))
    g0 = state_arr[:, :, 0, ...] if nm >= 1 else np.zeros((ns, nl) + state_arr.shape[3:], dtype=state_arr.dtype)
    g1 = state_arr[:, :, 1, ...] if nm >= 2 else np.zeros_like(g0)
    g2 = state_arr[:, :, 2, ...] if nm >= 3 else np.zeros_like(g0)
    particle_density = np.sum(Jl * g0, axis=1)
    particle_upar = np.sum(Jl * g1, axis=1)
    particle_uperp = sqrt_b[None, ...] * np.sum(JlB * g0, axis=1)
    particle_temp = np.sqrt(2.0, dtype=np.float32) * np.sum(Jl * g2, axis=1)
    return {
        "ParticleDensity": particle_density,
        "ParticleUpar": particle_upar,
        "ParticleUperp": particle_uperp,
        "ParticleTemp": particle_temp,
    }


def _condense_kx(arr: np.ndarray) -> np.ndarray:
    return _take_axis(arr, _gx_active_kx_indices(np.asarray(arr).shape[-1]), axis=-1)


def _condense_ky(arr: np.ndarray) -> np.ndarray:
    return _take_axis(arr, _gx_active_ky_indices(np.asarray(arr).shape[-1]), axis=-1)


def _condense_kykx(arr: np.ndarray) -> np.ndarray:
    out = _take_axis(arr, _gx_active_ky_indices(np.asarray(arr).shape[-2]), axis=-2)
    return _take_axis(out, _gx_active_kx_indices(np.asarray(arr).shape[-1]), axis=-1)


def _write_gx_geometry_group(group: Any, cfg: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    grid = build_spectral_grid(cfg.grid)
    geom = ensure_flux_tube_geometry_data(build_runtime_geometry(cfg), grid.z)
    theta = np.asarray(grid.z, dtype=np.float32)
    group.createVariable("bmag", "f4", ("theta",))[:] = np.asarray(geom.bmag_profile, dtype=np.float32)
    group.createVariable("bgrad", "f4", ("theta",))[:] = np.asarray(geom.bgrad_profile, dtype=np.float32)
    group.createVariable("gbdrift", "f4", ("theta",))[:] = np.asarray(geom.gb_profile, dtype=np.float32)
    group.createVariable("gbdrift0", "f4", ("theta",))[:] = np.asarray(geom.gb0_profile, dtype=np.float32)
    group.createVariable("cvdrift", "f4", ("theta",))[:] = np.asarray(geom.cv_profile, dtype=np.float32)
    group.createVariable("cvdrift0", "f4", ("theta",))[:] = np.asarray(geom.cv0_profile, dtype=np.float32)
    group.createVariable("gds2", "f4", ("theta",))[:] = np.asarray(geom.gds2_profile, dtype=np.float32)
    group.createVariable("gds21", "f4", ("theta",))[:] = np.asarray(geom.gds21_profile, dtype=np.float32)
    group.createVariable("gds22", "f4", ("theta",))[:] = np.asarray(geom.gds22_profile, dtype=np.float32)
    group.createVariable("grho", "f4", ("theta",))[:] = np.asarray(geom.grho_profile, dtype=np.float32)
    group.createVariable("jacobian", "f4", ("theta",))[:] = np.asarray(geom.jacobian_profile, dtype=np.float32)
    group.createVariable("gradpar", "f4", ())[:] = np.float32(geom.gradpar_value)
    group.createVariable("nperiod", "i4", ())[:] = np.int32(cfg.grid.nperiod if cfg.grid.nperiod is not None else 1)
    group.createVariable("q", "f4", ())[:] = np.float32(geom.q)
    group.createVariable("shat", "f4", ())[:] = np.float32(geom.s_hat)
    group.createVariable("shift", "f4", ())[:] = np.float32(getattr(cfg.geometry, "shift", 0.0))
    group.createVariable("rmaj", "f4", ())[:] = np.float32(geom.R0)
    group.createVariable("aminor", "f4", ())[:] = np.float32(geom.epsilon * geom.R0)
    group.createVariable("kxfac", "f4", ())[:] = np.float32(geom.kxfac)
    group.createVariable("drhodpsi", "f4", ())[:] = np.float32(1.0)
    group.createVariable("theta_scale", "f4", ())[:] = np.float32(geom.theta_scale)
    group.createVariable("nfp", "i4", ())[:] = np.int32(geom.nfp)
    group.createVariable("alpha", "f4", ())[:] = np.float32(geom.alpha)
    group.createVariable("zeta_center", "f4", ())[:] = np.float32(0.0)
    return theta, np.asarray(real_fft_ordered_kx(grid.kx), dtype=np.float32), np.asarray(real_fft_unique_ky(grid.ky), dtype=np.float32), geom


def _write_gx_inputs_group(group: Any, cfg: Any, geom: Any) -> None:
    group.createVariable("igeo", "i4", ())[:] = np.int32(0 if str(cfg.geometry.model).lower() == "miller" else 1)
    group.createVariable("slab", "i4", ())[:] = np.int32(1 if str(cfg.geometry.model).lower() == "slab" else 0)
    group.createVariable("const_curv", "i4", ())[:] = np.int32(0)
    group.createVariable("geofile_dum", "i4", ())[:] = np.int32(1 if getattr(cfg.geometry, "geometry_file", None) else 0)
    group.createVariable("drhodpsi", "f4", ())[:] = np.float32(1.0)
    group.createVariable("kxfac", "f4", ())[:] = np.float32(geom.kxfac)
    group.createVariable("Rmaj", "f4", ())[:] = np.float32(geom.R0)
    group.createVariable("shift", "f4", ())[:] = np.float32(getattr(cfg.geometry, "shift", 0.0))
    group.createVariable("eps", "f4", ())[:] = np.float32(geom.epsilon)
    group.createVariable("q", "f4", ())[:] = np.float32(geom.q)
    group.createVariable("shat", "f4", ())[:] = np.float32(geom.s_hat)
    group.createVariable("kappa", "f4", ())[:] = np.float32(getattr(cfg.geometry, "kappa", 1.0))
    group.createVariable("kappa_prime", "f4", ())[:] = np.float32(getattr(cfg.geometry, "akappri", 0.0))
    group.createVariable("tri", "f4", ())[:] = np.float32(getattr(cfg.geometry, "tri", 0.0))
    group.createVariable("tri_prime", "f4", ())[:] = np.float32(getattr(cfg.geometry, "tripri", 0.0))
    group.createVariable("beta", "f4", ())[:] = np.float32(cfg.physics.beta)
    group.createVariable("zero_shat", "i4", ())[:] = np.int32(abs(float(geom.s_hat)) < 1.0e-30)
    group.createVariable("B_ref", "f4", ())[:] = np.float32(geom.B0)
    group.createVariable("a_ref", "f4", ())[:] = np.float32(max(float(geom.epsilon * geom.R0), 1.0))
    group.createVariable("grhoavg", "f4", ())[:] = np.float32(np.mean(np.asarray(geom.grho_profile, dtype=np.float32)))
    group.createVariable("surfarea", "f4", ())[:] = np.float32(np.nan)


def _write_runtime_nonlinear_gx_artifacts(out: str | Path, result: Any, cfg: Any) -> dict[str, str]:
    Dataset = _require_netcdf4()
    out_path = Path(out)
    base = _gx_bundle_base(out_path)
    out_nc_path = Path(f"{base}.out.nc")
    restart_path = _resolve_restart_path(out_path, cfg, for_write=True)
    big_path = Path(f"{base}.big.nc")

    diag: SimulationDiagnostics | None = result.diagnostics
    if diag is None:
        raise ValueError("GX-style nonlinear NetCDF artifacts require nonlinear diagnostics output")

    grid = build_spectral_grid(cfg.grid)
    geom_data = ensure_flux_tube_geometry_data(build_runtime_geometry(cfg), grid.z)
    theta = np.asarray(grid.z, dtype=np.float32)
    kx_vals = _gx_active_kx_values(np.asarray(grid.kx))
    ky_vals = _gx_active_ky_values(np.asarray(grid.ky))
    nspecies = int(np.asarray(result.state).shape[0]) if result.state is not None and np.asarray(result.state).ndim == 6 else len(cfg.species)
    time_vals = np.asarray(diag.t, dtype=np.float64)
    x_vals = _real_space_axis(int(grid.kx.size), float(cfg.grid.Lx))
    y_extent = float(2.0 * np.pi * grid.y0)
    y_vals = _real_space_axis(int(grid.ky.size), y_extent)
    nl = int(np.asarray(result.state).shape[1]) if result.state is not None and np.asarray(result.state).ndim == 6 else 1
    nm = int(np.asarray(result.state).shape[2]) if result.state is not None and np.asarray(result.state).ndim == 6 else 1

    _ensure_parent(out_nc_path)
    with Dataset(out_nc_path, "w") as root:
        root.createDimension("ri", 2)
        root.createDimension("x", x_vals.size)
        root.createDimension("y", y_vals.size)
        root.createDimension("theta", theta.size)
        root.createDimension("kx", kx_vals.size)
        root.createDimension("ky", ky_vals.size)
        root.createDimension("kz", theta.size)
        root.createDimension("m", nm)
        root.createDimension("l", nl)
        root.createDimension("s", nspecies)
        root.createDimension("time", time_vals.size)
        _write_runtime_root_metadata(root, cfg, nspecies=nspecies, nl=nl, nm=nm)

        grids = root.createGroup("Grids")
        grids.createVariable("time", "f8", ("time",))[:] = time_vals
        grids.createVariable("kx", "f4", ("kx",))[:] = kx_vals
        grids.createVariable("ky", "f4", ("ky",))[:] = ky_vals
        grids.createVariable("kz", "f4", ("kz",))[:] = theta
        grids.createVariable("x", "f4", ("x",))[:] = x_vals
        grids.createVariable("y", "f4", ("y",))[:] = y_vals
        grids.createVariable("theta", "f4", ("theta",))[:] = theta

        geom_group = root.createGroup("Geometry")
        _write_gx_geometry_group(geom_group, cfg)

        diag_group = root.createGroup("Diagnostics")
        resolved = diag.resolved
        if resolved is not None and resolved.Phi2_kxt is not None:
            phi2_t = np.sum(np.asarray(resolved.Phi2_kxt, dtype=np.float32), axis=1)
        else:
            phi2_t = np.asarray(diag.Wphi_t, dtype=np.float32)
        diag_group.createVariable("Phi2_t", "f4", ("time",))[:] = phi2_t
        wg_s = _species_matrix(np.asarray(diag.Wg_t, dtype=np.float32), nspecies, None)
        wphi_s = _species_matrix(np.asarray(diag.Wphi_t, dtype=np.float32), nspecies, None)
        wapar_s = _species_matrix(np.asarray(diag.Wapar_t, dtype=np.float32), nspecies, None)
        heat_s = _species_matrix(np.asarray(diag.heat_flux_t, dtype=np.float32), nspecies, None if diag.heat_flux_species_t is None else np.asarray(diag.heat_flux_species_t, dtype=np.float32))
        pflux_s = _species_matrix(np.asarray(diag.particle_flux_t, dtype=np.float32), nspecies, None if diag.particle_flux_species_t is None else np.asarray(diag.particle_flux_species_t, dtype=np.float32))
        turb_heat_s = _species_matrix(
            np.asarray(np.zeros_like(diag.heat_flux_t) if diag.turbulent_heating_t is None else diag.turbulent_heating_t, dtype=np.float32),
            nspecies,
            None if diag.turbulent_heating_species_t is None else np.asarray(diag.turbulent_heating_species_t, dtype=np.float32),
        )
        diag_group.createVariable("Wg_st", "f4", ("time", "s"))[:, :] = wg_s
        diag_group.createVariable("Wphi_st", "f4", ("time", "s"))[:, :] = wphi_s
        diag_group.createVariable("Wapar_st", "f4", ("time", "s"))[:, :] = wapar_s
        diag_group.createVariable("HeatFlux_st", "f4", ("time", "s"))[:, :] = heat_s
        diag_group.createVariable("ParticleFlux_st", "f4", ("time", "s"))[:, :] = pflux_s
        heat_es_st = _resolved_species_time(
            None if resolved is None else resolved.HeatFluxES_kxst,
            fallback=heat_s if cfg.physics.electrostatic else np.zeros_like(heat_s),
        )
        heat_apar_st = _resolved_species_time(
            None if resolved is None else resolved.HeatFluxApar_kxst,
            fallback=np.zeros_like(heat_s),
        )
        heat_bpar_st = _resolved_species_time(
            None if resolved is None else resolved.HeatFluxBpar_kxst,
            fallback=np.zeros_like(heat_s),
        )
        pflux_es_st = _resolved_species_time(
            None if resolved is None else resolved.ParticleFluxES_kxst,
            fallback=pflux_s if cfg.physics.electrostatic else np.zeros_like(pflux_s),
        )
        pflux_apar_st = _resolved_species_time(
            None if resolved is None else resolved.ParticleFluxApar_kxst,
            fallback=np.zeros_like(pflux_s),
        )
        pflux_bpar_st = _resolved_species_time(
            None if resolved is None else resolved.ParticleFluxBpar_kxst,
            fallback=np.zeros_like(pflux_s),
        )
        diag_group.createVariable("HeatFluxES_st", "f4", ("time", "s"))[:, :] = heat_es_st
        diag_group.createVariable("HeatFluxApar_st", "f4", ("time", "s"))[:, :] = heat_apar_st
        diag_group.createVariable("HeatFluxBpar_st", "f4", ("time", "s"))[:, :] = heat_bpar_st
        diag_group.createVariable("ParticleFluxES_st", "f4", ("time", "s"))[:, :] = pflux_es_st
        diag_group.createVariable("ParticleFluxApar_st", "f4", ("time", "s"))[:, :] = pflux_apar_st
        diag_group.createVariable("ParticleFluxBpar_st", "f4", ("time", "s"))[:, :] = pflux_bpar_st
        turb_heat_st = _resolved_species_time(
            None if resolved is None else resolved.TurbulentHeating_kxst,
            fallback=turb_heat_s,
        )
        diag_group.createVariable("TurbulentHeating_st", "f4", ("time", "s"))[:, :] = turb_heat_st
        if resolved is not None:
            if resolved.Phi2_kxt is not None:
                diag_group.createVariable("Phi2_kxt", "f4", ("time", "kx"))[:, :] = _condense_kx(np.asarray(resolved.Phi2_kxt, dtype=np.float32))
            if resolved.Phi2_kyt is not None:
                diag_group.createVariable("Phi2_kyt", "f4", ("time", "ky"))[:, :] = _condense_ky(np.asarray(resolved.Phi2_kyt, dtype=np.float32))
            if resolved.Phi2_kxkyt is not None:
                diag_group.createVariable("Phi2_kxkyt", "f4", ("time", "ky", "kx"))[:, :, :] = _condense_kykx(np.asarray(resolved.Phi2_kxkyt, dtype=np.float32))
            if resolved.Phi2_zt is not None:
                diag_group.createVariable("Phi2_zt", "f4", ("time", "theta"))[:, :] = np.asarray(resolved.Phi2_zt, dtype=np.float32)
            if resolved.Phi2_zonal_t is not None:
                diag_group.createVariable("Phi2_zonal_t", "f4", ("time",))[:] = np.asarray(resolved.Phi2_zonal_t, dtype=np.float32)
            if resolved.Phi2_zonal_kxt is not None:
                diag_group.createVariable("Phi2_zonal_kxt", "f4", ("time", "kx"))[:, :] = _condense_kx(np.asarray(resolved.Phi2_zonal_kxt, dtype=np.float32))
            if resolved.Phi2_zonal_zt is not None:
                diag_group.createVariable("Phi2_zonal_zt", "f4", ("time", "theta"))[:, :] = np.asarray(resolved.Phi2_zonal_zt, dtype=np.float32)
            metric_specs = (
                ("Wg", resolved.Wg_kxst, resolved.Wg_kyst, resolved.Wg_kxkyst, resolved.Wg_zst),
                ("Wphi", resolved.Wphi_kxst, resolved.Wphi_kyst, resolved.Wphi_kxkyst, resolved.Wphi_zst),
                ("Wapar", resolved.Wapar_kxst, resolved.Wapar_kyst, resolved.Wapar_kxkyst, resolved.Wapar_zst),
                ("HeatFlux", resolved.HeatFlux_kxst, resolved.HeatFlux_kyst, resolved.HeatFlux_kxkyst, resolved.HeatFlux_zst),
                ("ParticleFlux", resolved.ParticleFlux_kxst, resolved.ParticleFlux_kyst, resolved.ParticleFlux_kxkyst, resolved.ParticleFlux_zst),
            )
            for prefix, kx_arr, ky_arr, kykx_arr, z_arr in metric_specs:
                if kx_arr is not None:
                    diag_group.createVariable(f"{prefix}_kxst", "f4", ("time", "s", "kx"))[:, :, :] = _condense_kx(np.asarray(kx_arr, dtype=np.float32))
                if ky_arr is not None:
                    diag_group.createVariable(f"{prefix}_kyst", "f4", ("time", "s", "ky"))[:, :, :] = _condense_ky(np.asarray(ky_arr, dtype=np.float32))
                if kykx_arr is not None:
                    diag_group.createVariable(f"{prefix}_kxkyst", "f4", ("time", "s", "ky", "kx"))[:, :, :, :] = _condense_kykx(np.asarray(kykx_arr, dtype=np.float32))
                if z_arr is not None:
                    diag_group.createVariable(f"{prefix}_zst", "f4", ("time", "s", "theta"))[:, :, :] = np.asarray(z_arr, dtype=np.float32)
            if resolved.Wg_lmst is not None:
                diag_group.createVariable("Wg_lmst", "f4", ("time", "s", "m", "l"))[:, :, :, :] = np.asarray(resolved.Wg_lmst, dtype=np.float32)
            split_metric_specs = (
                (
                    "HeatFluxES",
                    resolved.HeatFluxES_kxst,
                    resolved.HeatFluxES_kyst,
                    resolved.HeatFluxES_kxkyst,
                    resolved.HeatFluxES_zst,
                    resolved.HeatFlux_kxst,
                    resolved.HeatFlux_kyst,
                    resolved.HeatFlux_kxkyst,
                    resolved.HeatFlux_zst,
                    cfg.physics.electrostatic,
                ),
                (
                    "HeatFluxApar",
                    resolved.HeatFluxApar_kxst,
                    resolved.HeatFluxApar_kyst,
                    resolved.HeatFluxApar_kxkyst,
                    resolved.HeatFluxApar_zst,
                    None,
                    None,
                    None,
                    None,
                    False,
                ),
                (
                    "HeatFluxBpar",
                    resolved.HeatFluxBpar_kxst,
                    resolved.HeatFluxBpar_kyst,
                    resolved.HeatFluxBpar_kxkyst,
                    resolved.HeatFluxBpar_zst,
                    None,
                    None,
                    None,
                    None,
                    False,
                ),
                (
                    "ParticleFluxES",
                    resolved.ParticleFluxES_kxst,
                    resolved.ParticleFluxES_kyst,
                    resolved.ParticleFluxES_kxkyst,
                    resolved.ParticleFluxES_zst,
                    resolved.ParticleFlux_kxst,
                    resolved.ParticleFlux_kyst,
                    resolved.ParticleFlux_kxkyst,
                    resolved.ParticleFlux_zst,
                    cfg.physics.electrostatic,
                ),
                (
                    "ParticleFluxApar",
                    resolved.ParticleFluxApar_kxst,
                    resolved.ParticleFluxApar_kyst,
                    resolved.ParticleFluxApar_kxkyst,
                    resolved.ParticleFluxApar_zst,
                    None,
                    None,
                    None,
                    None,
                    False,
                ),
                (
                    "ParticleFluxBpar",
                    resolved.ParticleFluxBpar_kxst,
                    resolved.ParticleFluxBpar_kyst,
                    resolved.ParticleFluxBpar_kxkyst,
                    resolved.ParticleFluxBpar_zst,
                    None,
                    None,
                    None,
                    None,
                    False,
                ),
            )
            for prefix, kx_arr, ky_arr, kykx_arr, z_arr, total_kx, total_ky, total_kykx, total_z, fallback_total in split_metric_specs:
                use_kx = total_kx if kx_arr is None and fallback_total else kx_arr
                use_ky = total_ky if ky_arr is None and fallback_total else ky_arr
                use_kykx = total_kykx if kykx_arr is None and fallback_total else kykx_arr
                use_z = total_z if z_arr is None and fallback_total else z_arr
                if use_kx is not None:
                    diag_group.createVariable(f"{prefix}_kxst", "f4", ("time", "s", "kx"))[:, :, :] = _condense_kx(np.asarray(use_kx, dtype=np.float32))
                if use_ky is not None:
                    diag_group.createVariable(f"{prefix}_kyst", "f4", ("time", "s", "ky"))[:, :, :] = _condense_ky(np.asarray(use_ky, dtype=np.float32))
                if use_kykx is not None:
                    diag_group.createVariable(f"{prefix}_kxkyst", "f4", ("time", "s", "ky", "kx"))[:, :, :, :] = _condense_kykx(np.asarray(use_kykx, dtype=np.float32))
                if use_z is not None:
                    diag_group.createVariable(f"{prefix}_zst", "f4", ("time", "s", "theta"))[:, :, :] = np.asarray(use_z, dtype=np.float32)
            if resolved.TurbulentHeating_kxst is not None:
                diag_group.createVariable("TurbulentHeating_kxst", "f4", ("time", "s", "kx"))[:, :, :] = _condense_kx(np.asarray(resolved.TurbulentHeating_kxst, dtype=np.float32))
            if resolved.TurbulentHeating_kyst is not None:
                diag_group.createVariable("TurbulentHeating_kyst", "f4", ("time", "s", "ky"))[:, :, :] = _condense_ky(np.asarray(resolved.TurbulentHeating_kyst, dtype=np.float32))
            if resolved.TurbulentHeating_kxkyst is not None:
                diag_group.createVariable("TurbulentHeating_kxkyst", "f4", ("time", "s", "ky", "kx"))[:, :, :, :] = _condense_kykx(np.asarray(resolved.TurbulentHeating_kxkyst, dtype=np.float32))
            if resolved.TurbulentHeating_zst is not None:
                diag_group.createVariable("TurbulentHeating_zst", "f4", ("time", "s", "theta"))[:, :, :] = np.asarray(resolved.TurbulentHeating_zst, dtype=np.float32)

        inputs = root.createGroup("Inputs")
        _write_gx_inputs_group(inputs, cfg, geom_data)

    paths = {"out": str(out_nc_path)}

    if result.state is not None:
        gx_state = _restart_to_gx_layout(np.asarray(result.state))
        _ensure_parent(restart_path)
        with Dataset(restart_path, "w") as root:
            root.createDimension("Nspecies", gx_state.shape[0])
            root.createDimension("Nm", gx_state.shape[1])
            root.createDimension("Nl", gx_state.shape[2])
            root.createDimension("Nz", gx_state.shape[3])
            root.createDimension("Nkx", gx_state.shape[4])
            root.createDimension("Nky", gx_state.shape[5])
            root.createDimension("ri", 2)
            root.createVariable("G", "f4", ("Nspecies", "Nm", "Nl", "Nz", "Nkx", "Nky", "ri"))[:, :, :, :, :, :, :] = gx_state
            time_last = float(time_vals[-1]) if time_vals.size else 0.0
            root.createVariable("time", "f8", ())[:] = time_last
        paths["restart"] = str(restart_path)

    if result.fields is not None:
        _ensure_parent(big_path)
        phi_full = np.asarray(result.fields.phi)
        apar_full = np.zeros_like(phi_full) if result.fields.apar is None else np.asarray(result.fields.apar)
        bpar_full = np.zeros_like(phi_full) if result.fields.bpar is None else np.asarray(result.fields.bpar)
        phi_active = _gx_active_field(phi_full)
        apar_active = _gx_active_field(apar_full)
        bpar_active = _gx_active_field(bpar_full)
        basis_moments = _state_basis_moments(np.asarray(result.state)) if result.state is not None else {}
        particle_moments = _particle_moments(np.asarray(result.state), cfg) if result.state is not None else {}
        with Dataset(big_path, "w") as root:
            root.createDimension("ri", 2)
            root.createDimension("x", x_vals.size)
            root.createDimension("y", y_vals.size)
            root.createDimension("theta", theta.size)
            root.createDimension("kx", kx_vals.size)
            root.createDimension("ky", ky_vals.size)
            root.createDimension("kz", theta.size)
            root.createDimension("m", nm)
            root.createDimension("l", nl)
            root.createDimension("s", nspecies)
            root.createDimension("time", 1)
            _write_runtime_root_metadata(root, cfg, nspecies=nspecies, nl=nl, nm=nm)
            grids = root.createGroup("Grids")
            grids.createVariable("time", "f8", ("time",))[:] = np.asarray([float(time_vals[-1]) if time_vals.size else 0.0], dtype=np.float64)
            grids.createVariable("kx", "f4", ("kx",))[:] = kx_vals
            grids.createVariable("ky", "f4", ("ky",))[:] = ky_vals
            grids.createVariable("kz", "f4", ("kz",))[:] = theta
            grids.createVariable("x", "f4", ("x",))[:] = x_vals
            grids.createVariable("y", "f4", ("y",))[:] = y_vals
            grids.createVariable("theta", "f4", ("theta",))[:] = theta
            geom_group = root.createGroup("Geometry")
            _write_gx_geometry_group(geom_group, cfg)
            diag_group = root.createGroup("Diagnostics")
            diag_group.createVariable("Phi", "f4", ("time", "ky", "kx", "theta", "ri"))[0, ...] = _spectral_to_ri(phi_active)
            diag_group.createVariable("Apar", "f4", ("time", "ky", "kx", "theta", "ri"))[0, ...] = _spectral_to_ri(apar_active)
            diag_group.createVariable("Bpar", "f4", ("time", "ky", "kx", "theta", "ri"))[0, ...] = _spectral_to_ri(bpar_active)
            diag_group.createVariable("PhiXY", "f4", ("time", "y", "x", "theta"))[0, ...] = _spectral_to_xy(phi_full)
            diag_group.createVariable("AparXY", "f4", ("time", "y", "x", "theta"))[0, ...] = _spectral_to_xy(apar_full)
            diag_group.createVariable("BparXY", "f4", ("time", "y", "x", "theta"))[0, ...] = _spectral_to_xy(bpar_full)
            for name, values in basis_moments.items():
                active = _gx_active_field(values, ky_axis=1, kx_axis=2)
                diag_group.createVariable(name, "f4", ("time", "s", "ky", "kx", "theta", "ri"))[0, ...] = _spectral_species_to_ri(active)
                diag_group.createVariable(f"{name}XY", "f4", ("time", "s", "y", "x", "theta"))[0, ...] = np.real(np.fft.ifft2(values, axes=(1, 2))).astype(np.float32, copy=False)
            for name, values in particle_moments.items():
                active = _gx_active_field(values, ky_axis=1, kx_axis=2)
                diag_group.createVariable(name, "f4", ("time", "s", "ky", "kx", "theta", "ri"))[0, ...] = _spectral_species_to_ri(active)
                diag_group.createVariable(f"{name}XY", "f4", ("time", "s", "y", "x", "theta"))[0, ...] = np.real(np.fft.ifft2(values, axes=(1, 2))).astype(np.float32, copy=False)
        paths["big"] = str(big_path)

    return paths


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
        "fit_window_tmin": None if result.fit_window_tmin is None else float(result.fit_window_tmin),
        "fit_window_tmax": None if result.fit_window_tmax is None else float(result.fit_window_tmax),
        "fit_signal_used": result.fit_signal_used,
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

    state_path = _write_state(base, None if result.state is None else np.asarray(result.state))
    if state_path is not None:
        paths["state"] = str(state_path)
    return paths


def write_runtime_nonlinear_artifacts(out: str | Path, result: Any, cfg: Any | None = None) -> dict[str, str]:
    """Write summary/diagnostics/state artifacts for a nonlinear runtime run."""

    out_path = Path(out)
    if _is_gx_netcdf_target(out_path):
        if cfg is None:
            raise ValueError("cfg is required to write GX-style nonlinear NetCDF artifacts")
        return _write_runtime_nonlinear_gx_artifacts(out_path, result, cfg)

    base = _artifact_base(out_path)
    summary_path = out_path if out_path.suffix.lower() == ".json" else Path(f"{base}.summary.json")
    csv_path = out_path if out_path.suffix.lower() == ".csv" else Path(f"{base}.diagnostics.csv")

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

    state_path = _write_state(base, None if result.state is None else np.asarray(result.state))
    if state_path is not None:
        paths["state"] = str(state_path)
    return paths
