"""Geometry and particle-moment helpers for nonlinear NetCDF output."""

from __future__ import annotations

from typing import Any

import numpy as np

from spectraxgk.core.grid import (
    build_spectral_grid,
    real_fft_ordered_kx,
    real_fft_unique_ky,
)
from spectraxgk.geometry import (
    apply_geometry_grid_defaults,
    ensure_flux_tube_geometry_data,
)
from spectraxgk.operators.linear.cache import build_linear_cache
from spectraxgk.runtime import (
    build_runtime_geometry,
    build_runtime_linear_params,
)

def _build_output_grid_and_geometry(cfg: Any) -> tuple[Any, Any]:
    """Resolve artifact output onto the same geometry-implied grid as the solver."""

    geom_raw = build_runtime_geometry(cfg)
    grid_cfg = apply_geometry_grid_defaults(geom_raw, cfg.grid)
    grid = build_spectral_grid(grid_cfg)
    geom = ensure_flux_tube_geometry_data(geom_raw, grid.z)
    return grid, geom


def _particle_moments(state: np.ndarray, cfg: Any) -> dict[str, np.ndarray]:
    state_arr = np.asarray(state)
    ns, nl, nm, _ny, _nx, _nz = state_arr.shape
    grid, geom = _build_output_grid_and_geometry(cfg)
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


def _write_geometry_group(
    group: Any,
    cfg: Any,
    *,
    grid: Any | None = None,
    geom: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    if grid is None or geom is None:
        grid, geom = _build_output_grid_and_geometry(cfg)
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


def _write_input_parameters_group(group: Any, cfg: Any, geom: Any) -> None:
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



__all__ = [
    "_build_output_grid_and_geometry",
    "_particle_moments",
    "_write_geometry_group",
    "_write_input_parameters_group",
]
