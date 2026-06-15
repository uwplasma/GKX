"""Structured runtime artifact writers for the executable and benchmark tooling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from spectraxgk.geometry import (
    apply_geometry_grid_defaults,
    ensure_flux_tube_geometry_data,
)
from spectraxgk.grids import (
    build_spectral_grid,
    real_fft_ordered_kx,
    real_fft_unique_ky,
)
from spectraxgk.diagnostics import (
    SimulationDiagnostics,
)
from spectraxgk.linear import build_linear_cache
from spectraxgk.runtime import (
    RuntimeNonlinearResult,
    _concat_gx_diagnostics,
    build_runtime_geometry,
    build_runtime_linear_params,
    run_runtime_nonlinear,
)
from spectraxgk.runtime_artifact_diagnostics import (
    validate_finite_runtime_result as _validate_finite_runtime_result,
)
from spectraxgk.runtime_orchestration import (
    RuntimeArtifactHandoffDeps,
    run_runtime_nonlinear_artifact_handoff,
)

from spectraxgk.runtime_artifact_gx_layout import (
    _complex_to_ri,
    _condense_kx as _condense_kx,
    _condense_kx_for_output,
    _condense_ky as _condense_ky,
    _condense_ky_for_output,
    _condense_kykx as _condense_kykx,
    _condense_kykx_for_output,
    _gx_active_field,
    _gx_active_kx_count as _gx_active_kx_count,
    _gx_active_kx_indices as _gx_active_kx_indices,
    _gx_active_kx_values,
    _gx_active_ky_count as _gx_active_ky_count,
    _gx_active_ky_indices as _gx_active_ky_indices,
    _gx_active_ky_values,
    _maybe_var as _maybe_var,
    _real_space_axis,
    _require_netcdf4,
    _restart_to_gx_layout,
    _species_matrix,
    _spectral_species_to_ri,
    _spectral_to_ri,
    _spectral_to_xy,
    _state_basis_moments,
    _take_axis as _take_axis,
    _write_runtime_root_metadata,
)
from spectraxgk.runtime_artifact_nonlinear_diagnostics import (
    _condense_gx_diagnostics_for_output as _condense_gx_diagnostics_for_output,
    _condense_resolved_for_output as _condense_resolved_for_output,
    _read_optional_var as _read_optional_var,
    _resolved_species_time as _resolved_species_time,
    _resolve_restart_path as _resolve_restart_path,
    load_runtime_nonlinear_gx_diagnostics as load_runtime_nonlinear_gx_diagnostics,
)
from spectraxgk.runtime_artifact_io import (
    _artifact_base,
    _ensure_parent,
    _flatten_series,
    _gx_bundle_base,
    _is_gx_netcdf_target,
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
    deps = RuntimeArtifactHandoffDeps(
        is_gx_netcdf_target=_is_gx_netcdf_target,
        resolve_restart_path=lambda path, run_cfg: _resolve_restart_path(
            path, run_cfg, for_write=False
        ),
        resolve_restart_write_path=lambda path, run_cfg: _resolve_restart_path(
            path, run_cfg, for_write=True
        ),
        gx_bundle_base=_gx_bundle_base,
        load_runtime_nonlinear_gx_diagnostics=load_runtime_nonlinear_gx_diagnostics,
        condense_gx_diagnostics_for_output=_condense_gx_diagnostics_for_output,
        concat_gx_diagnostics=_concat_gx_diagnostics,
        validate_finite_runtime_result=lambda result: _validate_finite_runtime_result(
            result, label="nonlinear runtime chunk"
        ),
        run_runtime_nonlinear=run_runtime_nonlinear,
        write_runtime_nonlinear_artifacts=write_runtime_nonlinear_artifacts,
    )
    return run_runtime_nonlinear_artifact_handoff(
        cfg,
        out=out,
        ky_target=ky_target,
        kx_target=kx_target,
        Nl=Nl,
        Nm=Nm,
        dt=dt,
        steps=steps,
        method=method,
        sample_stride=sample_stride,
        diagnostics_stride=diagnostics_stride,
        laguerre_mode=laguerre_mode,
        diagnostics=diagnostics,
        show_progress=show_progress,
        status_callback=status_callback,
        deps=deps,
    )


def _build_artifact_grid_and_geometry(cfg: Any) -> tuple[Any, Any]:
    """Resolve artifact output onto the same geometry-implied grid as the solver."""

    geom_raw = build_runtime_geometry(cfg)
    grid_cfg = apply_geometry_grid_defaults(geom_raw, cfg.grid)
    grid = build_spectral_grid(grid_cfg)
    geom = ensure_flux_tube_geometry_data(geom_raw, grid.z)
    return grid, geom


def _particle_moments(state: np.ndarray, cfg: Any) -> dict[str, np.ndarray]:
    state_arr = np.asarray(state)
    ns, nl, nm, _ny, _nx, _nz = state_arr.shape
    grid, geom = _build_artifact_grid_and_geometry(cfg)
    params = build_runtime_linear_params(cfg, Nm=nm, geom=geom)
    cache = build_linear_cache(grid, geom, params, nl, nm)
    Jl = np.asarray(cache.Jl)
    JlB = np.asarray(cache.JlB)
    if Jl.ndim == 4:
        Jl = Jl[None, ...]
    if JlB.ndim == 4:
        JlB = JlB[None, ...]
    sqrt_b = np.sqrt(np.maximum(np.asarray(cache.kperp2, dtype=np.float32), 0.0))
    g0 = (
        state_arr[:, :, 0, ...]
        if nm >= 1
        else np.zeros((ns, nl) + state_arr.shape[3:], dtype=state_arr.dtype)
    )
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


def _write_gx_geometry_group(
    group: Any,
    cfg: Any,
    *,
    grid: Any | None = None,
    geom: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    if grid is None or geom is None:
        grid, geom = _build_artifact_grid_and_geometry(cfg)
    theta = np.asarray(grid.z, dtype=np.float32)
    group.createVariable("bmag", "f4", ("theta",))[:] = np.asarray(
        geom.bmag_profile, dtype=np.float32
    )
    group.createVariable("bgrad", "f4", ("theta",))[:] = np.asarray(
        geom.bgrad_profile, dtype=np.float32
    )
    group.createVariable("gbdrift", "f4", ("theta",))[:] = np.asarray(
        geom.gb_profile, dtype=np.float32
    )
    group.createVariable("gbdrift0", "f4", ("theta",))[:] = np.asarray(
        geom.gb0_profile, dtype=np.float32
    )
    group.createVariable("cvdrift", "f4", ("theta",))[:] = np.asarray(
        geom.cv_profile, dtype=np.float32
    )
    group.createVariable("cvdrift0", "f4", ("theta",))[:] = np.asarray(
        geom.cv0_profile, dtype=np.float32
    )
    group.createVariable("gds2", "f4", ("theta",))[:] = np.asarray(
        geom.gds2_profile, dtype=np.float32
    )
    group.createVariable("gds21", "f4", ("theta",))[:] = np.asarray(
        geom.gds21_profile, dtype=np.float32
    )
    group.createVariable("gds22", "f4", ("theta",))[:] = np.asarray(
        geom.gds22_profile, dtype=np.float32
    )
    group.createVariable("grho", "f4", ("theta",))[:] = np.asarray(
        geom.grho_profile, dtype=np.float32
    )
    group.createVariable("jacobian", "f4", ("theta",))[:] = np.asarray(
        geom.jacobian_profile, dtype=np.float32
    )
    group.createVariable("gradpar", "f4", ())[:] = np.float32(geom.gradpar_value)
    group.createVariable("nperiod", "i4", ())[:] = np.int32(
        cfg.grid.nperiod if cfg.grid.nperiod is not None else 1
    )
    group.createVariable("q", "f4", ())[:] = np.float32(geom.q)
    group.createVariable("shat", "f4", ())[:] = np.float32(geom.s_hat)
    group.createVariable("shift", "f4", ())[:] = np.float32(
        getattr(cfg.geometry, "shift", 0.0)
    )
    group.createVariable("rmaj", "f4", ())[:] = np.float32(geom.R0)
    group.createVariable("aminor", "f4", ())[:] = np.float32(geom.epsilon * geom.R0)
    group.createVariable("kxfac", "f4", ())[:] = np.float32(geom.kxfac)
    group.createVariable("drhodpsi", "f4", ())[:] = np.float32(1.0)
    group.createVariable("theta_scale", "f4", ())[:] = np.float32(geom.theta_scale)
    group.createVariable("nfp", "i4", ())[:] = np.int32(geom.nfp)
    group.createVariable("alpha", "f4", ())[:] = np.float32(geom.alpha)
    group.createVariable("zeta_center", "f4", ())[:] = np.float32(0.0)
    return (
        theta,
        np.asarray(real_fft_ordered_kx(grid.kx), dtype=np.float32),
        np.asarray(real_fft_unique_ky(grid.ky), dtype=np.float32),
        geom,
    )


def _write_gx_inputs_group(group: Any, cfg: Any, geom: Any) -> None:
    group.createVariable("igeo", "i4", ())[:] = np.int32(
        0 if str(cfg.geometry.model).lower() == "miller" else 1
    )
    group.createVariable("slab", "i4", ())[:] = np.int32(
        1 if str(cfg.geometry.model).lower() == "slab" else 0
    )
    group.createVariable("const_curv", "i4", ())[:] = np.int32(0)
    group.createVariable("geofile_dum", "i4", ())[:] = np.int32(
        1 if getattr(cfg.geometry, "geometry_file", None) else 0
    )
    group.createVariable("drhodpsi", "f4", ())[:] = np.float32(1.0)
    group.createVariable("kxfac", "f4", ())[:] = np.float32(geom.kxfac)
    group.createVariable("Rmaj", "f4", ())[:] = np.float32(geom.R0)
    group.createVariable("shift", "f4", ())[:] = np.float32(
        getattr(cfg.geometry, "shift", 0.0)
    )
    group.createVariable("eps", "f4", ())[:] = np.float32(geom.epsilon)
    group.createVariable("q", "f4", ())[:] = np.float32(geom.q)
    group.createVariable("shat", "f4", ())[:] = np.float32(geom.s_hat)
    group.createVariable("kappa", "f4", ())[:] = np.float32(
        getattr(cfg.geometry, "kappa", 1.0)
    )
    group.createVariable("kappa_prime", "f4", ())[:] = np.float32(
        getattr(cfg.geometry, "akappri", 0.0)
    )
    group.createVariable("tri", "f4", ())[:] = np.float32(
        getattr(cfg.geometry, "tri", 0.0)
    )
    group.createVariable("tri_prime", "f4", ())[:] = np.float32(
        getattr(cfg.geometry, "tripri", 0.0)
    )
    group.createVariable("beta", "f4", ())[:] = np.float32(cfg.physics.beta)
    group.createVariable("zero_shat", "i4", ())[:] = np.int32(
        abs(float(geom.s_hat)) < 1.0e-30
    )
    group.createVariable("B_ref", "f4", ())[:] = np.float32(geom.B0)
    group.createVariable("a_ref", "f4", ())[:] = np.float32(
        max(float(geom.epsilon * geom.R0), 1.0)
    )
    group.createVariable("grhoavg", "f4", ())[:] = np.float32(
        np.mean(np.asarray(geom.grho_profile, dtype=np.float32))
    )
    group.createVariable("surfarea", "f4", ())[:] = np.float32(np.nan)


def _write_runtime_nonlinear_gx_artifacts(
    out: str | Path, result: Any, cfg: Any
) -> dict[str, str]:
    Dataset = _require_netcdf4()
    out_path = Path(out)
    base = _gx_bundle_base(out_path)
    out_nc_path = Path(f"{base}.out.nc")
    restart_path = _resolve_restart_path(out_path, cfg, for_write=True)
    big_path = Path(f"{base}.big.nc")

    diag: SimulationDiagnostics | None = result.diagnostics
    if diag is None:
        raise ValueError(
            "GX-style nonlinear NetCDF artifacts require nonlinear diagnostics output"
        )

    grid, geom_data = _build_artifact_grid_and_geometry(cfg)
    theta = np.asarray(grid.z, dtype=np.float32)
    kx_vals = _gx_active_kx_values(np.asarray(grid.kx))
    ky_vals = _gx_active_ky_values(np.asarray(grid.ky))
    full_nx = int(np.asarray(grid.kx).size)
    full_ny = int(np.asarray(grid.ky).size)
    active_nx = int(kx_vals.size)
    active_ny = int(ky_vals.size)
    nspecies = (
        int(np.asarray(result.state).shape[0])
        if result.state is not None and np.asarray(result.state).ndim == 6
        else len(cfg.species)
    )
    time_vals = np.asarray(diag.t, dtype=np.float64)
    x_vals = _real_space_axis(int(grid.kx.size), float(2.0 * np.pi * grid.x0))
    y_extent = float(2.0 * np.pi * grid.y0)
    y_vals = _real_space_axis(int(grid.ky.size), y_extent)
    nl = (
        int(np.asarray(result.state).shape[1])
        if result.state is not None and np.asarray(result.state).ndim == 6
        else 1
    )
    nm = (
        int(np.asarray(result.state).shape[2])
        if result.state is not None and np.asarray(result.state).ndim == 6
        else 1
    )

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
        _write_gx_geometry_group(geom_group, cfg, grid=grid, geom=geom_data)

        diag_group = root.createGroup("Diagnostics")
        resolved = diag.resolved
        phi2_kx_out = None
        phi2_ky_out = None
        phi2_kykx_out = None
        if resolved is not None and resolved.Phi2_kxkyt is not None:
            # GX stores Phi2 on the rFFT-positive ky view.  Deriving the one-
            # dimensional spectra from the condensed two-dimensional spectrum
            # keeps Phi2_t, Phi2_kxt, and Phi2_kyt mutually consistent when
            # SPECTRAX-GK evolved a full Hermitian ky layout internally.
            phi2_kykx_out = _condense_kykx_for_output(
                np.asarray(resolved.Phi2_kxkyt, dtype=np.float32),
                full_ny=full_ny,
                full_nx=full_nx,
                active_ny=active_ny,
                active_nx=active_nx,
            )
            phi2_kx_out = np.sum(phi2_kykx_out, axis=1)
            phi2_ky_out = np.sum(phi2_kykx_out, axis=2)
            phi2_t = np.sum(phi2_kykx_out, axis=(1, 2))
        elif resolved is not None and resolved.Phi2_kyt is not None:
            phi2_ky_out = _condense_ky_for_output(
                np.asarray(resolved.Phi2_kyt, dtype=np.float32),
                full_ny=full_ny,
                active_ny=active_ny,
            )
            phi2_t = np.sum(phi2_ky_out, axis=1)
        elif resolved is not None and resolved.Phi2_kxt is not None:
            phi2_kx_out = _condense_kx_for_output(
                np.asarray(resolved.Phi2_kxt, dtype=np.float32),
                full_nx=full_nx,
                active_nx=active_nx,
            )
            phi2_t = np.sum(phi2_kx_out, axis=1)
        else:
            phi2_t = np.asarray(diag.Wphi_t, dtype=np.float32)
        diag_group.createVariable("Phi2_t", "f4", ("time",))[:] = phi2_t
        wg_s = _species_matrix(np.asarray(diag.Wg_t, dtype=np.float32), nspecies, None)
        wphi_s = _species_matrix(
            np.asarray(diag.Wphi_t, dtype=np.float32), nspecies, None
        )
        wapar_s = _species_matrix(
            np.asarray(diag.Wapar_t, dtype=np.float32), nspecies, None
        )
        heat_s = _species_matrix(
            np.asarray(diag.heat_flux_t, dtype=np.float32),
            nspecies,
            None
            if diag.heat_flux_species_t is None
            else np.asarray(diag.heat_flux_species_t, dtype=np.float32),
        )
        pflux_s = _species_matrix(
            np.asarray(diag.particle_flux_t, dtype=np.float32),
            nspecies,
            None
            if diag.particle_flux_species_t is None
            else np.asarray(diag.particle_flux_species_t, dtype=np.float32),
        )
        turb_heat_s = _species_matrix(
            np.asarray(
                np.zeros_like(diag.heat_flux_t)
                if diag.turbulent_heating_t is None
                else diag.turbulent_heating_t,
                dtype=np.float32,
            ),
            nspecies,
            None
            if diag.turbulent_heating_species_t is None
            else np.asarray(diag.turbulent_heating_species_t, dtype=np.float32),
        )
        diag_group.createVariable("Wg_st", "f4", ("time", "s"))[:, :] = wg_s
        diag_group.createVariable("Wphi_st", "f4", ("time", "s"))[:, :] = wphi_s
        diag_group.createVariable("Wapar_st", "f4", ("time", "s"))[:, :] = wapar_s
        diag_group.createVariable("HeatFlux_st", "f4", ("time", "s"))[:, :] = heat_s
        diag_group.createVariable("ParticleFlux_st", "f4", ("time", "s"))[:, :] = (
            pflux_s
        )
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
        diag_group.createVariable("HeatFluxES_st", "f4", ("time", "s"))[:, :] = (
            heat_es_st
        )
        diag_group.createVariable("HeatFluxApar_st", "f4", ("time", "s"))[:, :] = (
            heat_apar_st
        )
        diag_group.createVariable("HeatFluxBpar_st", "f4", ("time", "s"))[:, :] = (
            heat_bpar_st
        )
        diag_group.createVariable("ParticleFluxES_st", "f4", ("time", "s"))[:, :] = (
            pflux_es_st
        )
        diag_group.createVariable("ParticleFluxApar_st", "f4", ("time", "s"))[:, :] = (
            pflux_apar_st
        )
        diag_group.createVariable("ParticleFluxBpar_st", "f4", ("time", "s"))[:, :] = (
            pflux_bpar_st
        )
        turb_heat_st = _resolved_species_time(
            None if resolved is None else resolved.TurbulentHeating_kxst,
            fallback=turb_heat_s,
        )
        diag_group.createVariable("TurbulentHeating_st", "f4", ("time", "s"))[:, :] = (
            turb_heat_st
        )
        if resolved is not None:
            if phi2_kx_out is not None:
                diag_group.createVariable("Phi2_kxt", "f4", ("time", "kx"))[:, :] = (
                    phi2_kx_out
                )
            if phi2_ky_out is not None:
                diag_group.createVariable("Phi2_kyt", "f4", ("time", "ky"))[:, :] = (
                    phi2_ky_out
                )
            if phi2_kykx_out is not None:
                diag_group.createVariable("Phi2_kxkyt", "f4", ("time", "ky", "kx"))[
                    :, :, :
                ] = phi2_kykx_out
            if resolved.Phi2_zt is not None:
                diag_group.createVariable("Phi2_zt", "f4", ("time", "theta"))[:, :] = (
                    np.asarray(resolved.Phi2_zt, dtype=np.float32)
                )
            if resolved.Phi2_zonal_t is not None:
                diag_group.createVariable("Phi2_zonal_t", "f4", ("time",))[:] = (
                    np.asarray(resolved.Phi2_zonal_t, dtype=np.float32)
                )
            if resolved.Phi2_zonal_kxt is not None:
                diag_group.createVariable("Phi2_zonal_kxt", "f4", ("time", "kx"))[
                    :, :
                ] = _condense_kx_for_output(
                    np.asarray(resolved.Phi2_zonal_kxt, dtype=np.float32),
                    full_nx=full_nx,
                    active_nx=active_nx,
                )
            if resolved.Phi2_zonal_zt is not None:
                diag_group.createVariable("Phi2_zonal_zt", "f4", ("time", "theta"))[
                    :, :
                ] = np.asarray(resolved.Phi2_zonal_zt, dtype=np.float32)
            if resolved.Phi_zonal_mode_kxt is not None:
                diag_group.createVariable(
                    "Phi_zonal_mode_kxt", "f4", ("time", "kx", "ri")
                )[:, :, :] = _complex_to_ri(
                    _condense_kx_for_output(
                        np.asarray(resolved.Phi_zonal_mode_kxt),
                        full_nx=full_nx,
                        active_nx=active_nx,
                    )
                )
            if resolved.Phi_zonal_line_kxt is not None:
                diag_group.createVariable(
                    "Phi_zonal_line_kxt", "f4", ("time", "kx", "ri")
                )[:, :, :] = _complex_to_ri(
                    _condense_kx_for_output(
                        np.asarray(resolved.Phi_zonal_line_kxt),
                        full_nx=full_nx,
                        active_nx=active_nx,
                    )
                )
            metric_specs = (
                (
                    "Wg",
                    resolved.Wg_kxst,
                    resolved.Wg_kyst,
                    resolved.Wg_kxkyst,
                    resolved.Wg_zst,
                ),
                (
                    "Wphi",
                    resolved.Wphi_kxst,
                    resolved.Wphi_kyst,
                    resolved.Wphi_kxkyst,
                    resolved.Wphi_zst,
                ),
                (
                    "Wapar",
                    resolved.Wapar_kxst,
                    resolved.Wapar_kyst,
                    resolved.Wapar_kxkyst,
                    resolved.Wapar_zst,
                ),
                (
                    "HeatFlux",
                    resolved.HeatFlux_kxst,
                    resolved.HeatFlux_kyst,
                    resolved.HeatFlux_kxkyst,
                    resolved.HeatFlux_zst,
                ),
                (
                    "ParticleFlux",
                    resolved.ParticleFlux_kxst,
                    resolved.ParticleFlux_kyst,
                    resolved.ParticleFlux_kxkyst,
                    resolved.ParticleFlux_zst,
                ),
            )
            for prefix, kx_arr, ky_arr, kykx_arr, z_arr in metric_specs:
                if kx_arr is not None:
                    diag_group.createVariable(
                        f"{prefix}_kxst", "f4", ("time", "s", "kx")
                    )[:, :, :] = _condense_kx_for_output(
                        np.asarray(kx_arr, dtype=np.float32),
                        full_nx=full_nx,
                        active_nx=active_nx,
                    )
                if ky_arr is not None:
                    diag_group.createVariable(
                        f"{prefix}_kyst", "f4", ("time", "s", "ky")
                    )[:, :, :] = _condense_ky_for_output(
                        np.asarray(ky_arr, dtype=np.float32),
                        full_ny=full_ny,
                        active_ny=active_ny,
                    )
                if kykx_arr is not None:
                    diag_group.createVariable(
                        f"{prefix}_kxkyst", "f4", ("time", "s", "ky", "kx")
                    )[:, :, :, :] = _condense_kykx_for_output(
                        np.asarray(kykx_arr, dtype=np.float32),
                        full_ny=full_ny,
                        full_nx=full_nx,
                        active_ny=active_ny,
                        active_nx=active_nx,
                    )
                if z_arr is not None:
                    diag_group.createVariable(
                        f"{prefix}_zst", "f4", ("time", "s", "theta")
                    )[:, :, :] = np.asarray(z_arr, dtype=np.float32)
            if resolved.Wg_lmst is not None:
                diag_group.createVariable("Wg_lmst", "f4", ("time", "s", "m", "l"))[
                    :, :, :, :
                ] = np.asarray(resolved.Wg_lmst, dtype=np.float32)
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
            for (
                prefix,
                kx_arr,
                ky_arr,
                kykx_arr,
                z_arr,
                total_kx,
                total_ky,
                total_kykx,
                total_z,
                fallback_total,
            ) in split_metric_specs:
                use_kx = total_kx if kx_arr is None and fallback_total else kx_arr
                use_ky = total_ky if ky_arr is None and fallback_total else ky_arr
                use_kykx = (
                    total_kykx if kykx_arr is None and fallback_total else kykx_arr
                )
                use_z = total_z if z_arr is None and fallback_total else z_arr
                if use_kx is not None:
                    diag_group.createVariable(
                        f"{prefix}_kxst", "f4", ("time", "s", "kx")
                    )[:, :, :] = _condense_kx_for_output(
                        np.asarray(use_kx, dtype=np.float32),
                        full_nx=full_nx,
                        active_nx=active_nx,
                    )
                if use_ky is not None:
                    diag_group.createVariable(
                        f"{prefix}_kyst", "f4", ("time", "s", "ky")
                    )[:, :, :] = _condense_ky_for_output(
                        np.asarray(use_ky, dtype=np.float32),
                        full_ny=full_ny,
                        active_ny=active_ny,
                    )
                if use_kykx is not None:
                    diag_group.createVariable(
                        f"{prefix}_kxkyst", "f4", ("time", "s", "ky", "kx")
                    )[:, :, :, :] = _condense_kykx_for_output(
                        np.asarray(use_kykx, dtype=np.float32),
                        full_ny=full_ny,
                        full_nx=full_nx,
                        active_ny=active_ny,
                        active_nx=active_nx,
                    )
                if use_z is not None:
                    diag_group.createVariable(
                        f"{prefix}_zst", "f4", ("time", "s", "theta")
                    )[:, :, :] = np.asarray(use_z, dtype=np.float32)
            if resolved.TurbulentHeating_kxst is not None:
                diag_group.createVariable(
                    "TurbulentHeating_kxst", "f4", ("time", "s", "kx")
                )[:, :, :] = _condense_kx_for_output(
                    np.asarray(resolved.TurbulentHeating_kxst, dtype=np.float32),
                    full_nx=full_nx,
                    active_nx=active_nx,
                )
            if resolved.TurbulentHeating_kyst is not None:
                diag_group.createVariable(
                    "TurbulentHeating_kyst", "f4", ("time", "s", "ky")
                )[:, :, :] = _condense_ky_for_output(
                    np.asarray(resolved.TurbulentHeating_kyst, dtype=np.float32),
                    full_ny=full_ny,
                    active_ny=active_ny,
                )
            if resolved.TurbulentHeating_kxkyst is not None:
                diag_group.createVariable(
                    "TurbulentHeating_kxkyst", "f4", ("time", "s", "ky", "kx")
                )[:, :, :, :] = _condense_kykx_for_output(
                    np.asarray(resolved.TurbulentHeating_kxkyst, dtype=np.float32),
                    full_ny=full_ny,
                    full_nx=full_nx,
                    active_ny=active_ny,
                    active_nx=active_nx,
                )
            if resolved.TurbulentHeating_zst is not None:
                diag_group.createVariable(
                    "TurbulentHeating_zst", "f4", ("time", "s", "theta")
                )[:, :, :] = np.asarray(resolved.TurbulentHeating_zst, dtype=np.float32)

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
            root.createVariable(
                "G", "f4", ("Nspecies", "Nm", "Nl", "Nz", "Nkx", "Nky", "ri")
            )[:, :, :, :, :, :, :] = gx_state
            time_last = float(time_vals[-1]) if time_vals.size else 0.0
            root.createVariable("time", "f8", ())[:] = time_last
        paths["restart"] = str(restart_path)

    if result.fields is not None:
        _ensure_parent(big_path)
        phi_full = np.asarray(result.fields.phi)
        apar_full = (
            np.zeros_like(phi_full)
            if result.fields.apar is None
            else np.asarray(result.fields.apar)
        )
        bpar_full = (
            np.zeros_like(phi_full)
            if result.fields.bpar is None
            else np.asarray(result.fields.bpar)
        )
        phi_active = _gx_active_field(phi_full)
        apar_active = _gx_active_field(apar_full)
        bpar_active = _gx_active_field(bpar_full)
        basis_moments = (
            _state_basis_moments(np.asarray(result.state))
            if result.state is not None
            else {}
        )
        particle_moments = (
            _particle_moments(np.asarray(result.state), cfg)
            if result.state is not None
            else {}
        )
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
            grids.createVariable("time", "f8", ("time",))[:] = np.asarray(
                [float(time_vals[-1]) if time_vals.size else 0.0], dtype=np.float64
            )
            grids.createVariable("kx", "f4", ("kx",))[:] = kx_vals
            grids.createVariable("ky", "f4", ("ky",))[:] = ky_vals
            grids.createVariable("kz", "f4", ("kz",))[:] = theta
            grids.createVariable("x", "f4", ("x",))[:] = x_vals
            grids.createVariable("y", "f4", ("y",))[:] = y_vals
            grids.createVariable("theta", "f4", ("theta",))[:] = theta
            geom_group = root.createGroup("Geometry")
            _write_gx_geometry_group(geom_group, cfg)
            diag_group = root.createGroup("Diagnostics")
            diag_group.createVariable("Phi", "f4", ("time", "ky", "kx", "theta", "ri"))[
                0, ...
            ] = _spectral_to_ri(phi_active)
            diag_group.createVariable(
                "Apar", "f4", ("time", "ky", "kx", "theta", "ri")
            )[0, ...] = _spectral_to_ri(apar_active)
            diag_group.createVariable(
                "Bpar", "f4", ("time", "ky", "kx", "theta", "ri")
            )[0, ...] = _spectral_to_ri(bpar_active)
            diag_group.createVariable("PhiXY", "f4", ("time", "y", "x", "theta"))[
                0, ...
            ] = _spectral_to_xy(phi_full)
            diag_group.createVariable("AparXY", "f4", ("time", "y", "x", "theta"))[
                0, ...
            ] = _spectral_to_xy(apar_full)
            diag_group.createVariable("BparXY", "f4", ("time", "y", "x", "theta"))[
                0, ...
            ] = _spectral_to_xy(bpar_full)
            for name, values in basis_moments.items():
                active = _gx_active_field(values, ky_axis=1, kx_axis=2)
                diag_group.createVariable(
                    name, "f4", ("time", "s", "ky", "kx", "theta", "ri")
                )[0, ...] = _spectral_species_to_ri(active)
                diag_group.createVariable(
                    f"{name}XY", "f4", ("time", "s", "y", "x", "theta")
                )[0, ...] = np.real(np.fft.ifft2(values, axes=(1, 2))).astype(
                    np.float32, copy=False
                )
            for name, values in particle_moments.items():
                active = _gx_active_field(values, ky_axis=1, kx_axis=2)
                diag_group.createVariable(
                    name, "f4", ("time", "s", "ky", "kx", "theta", "ri")
                )[0, ...] = _spectral_species_to_ri(active)
                diag_group.createVariable(
                    f"{name}XY", "f4", ("time", "s", "y", "x", "theta")
                )[0, ...] = np.real(np.fft.ifft2(values, axes=(1, 2))).astype(
                    np.float32, copy=False
                )
        paths["big"] = str(big_path)

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


def write_runtime_nonlinear_artifacts(
    out: str | Path, result: Any, cfg: Any | None = None
) -> dict[str, str]:
    """Write summary/diagnostics/state artifacts for a nonlinear runtime run."""

    out_path = Path(out)
    if _is_gx_netcdf_target(out_path):
        if cfg is None:
            raise ValueError(
                "cfg is required to write GX-style nonlinear NetCDF artifacts"
            )
        return _write_runtime_nonlinear_gx_artifacts(out_path, result, cfg)

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
