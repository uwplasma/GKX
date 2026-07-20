"""Flux-tube geometry builders and grid-default policy."""

from __future__ import annotations

from dataclasses import replace
import math

import jax.numpy as jnp
import numpy as np

from gkx.config import GeometryConfig, GridConfig
from gkx.geometry.analytic import (
    SAlphaGeometry,
    SlabGeometry,
    ZERO_SHAT_THRESHOLD,
    effective_boundary,
    zero_shear_enabled,
)
from gkx.geometry.flux_tube import (
    FluxTubeGeometryData,
    _bgrad_from_bmag,
    _periodic_spectral_derivative,
    load_imported_geometry_netcdf,
    sample_flux_tube_geometry,
)

__all__ = [
    "FluxTubeGeometryData",
    "FluxTubeGeometryLike",
    "GeometryConfig",
    "GridConfig",
    "SAlphaGeometry",
    "SlabGeometry",
    "ZERO_SHAT_THRESHOLD",
    "_bgrad_from_bmag",
    "_periodic_spectral_derivative",
    "apply_geometry_grid_defaults",
    "apply_imported_geometry_grid_defaults",
    "build_flux_tube_geometry",
    "effective_boundary",
    "ensure_flux_tube_geometry_data",
    "load_imported_geometry_netcdf",
    "sample_flux_tube_geometry",
    "twist_shift_params",
    "zero_shear_enabled",
]

FluxTubeGeometryLike = SAlphaGeometry | SlabGeometry | FluxTubeGeometryData


def build_flux_tube_geometry(cfg: GeometryConfig) -> FluxTubeGeometryLike:
    """Build an analytic or imported flux-tube geometry from config."""

    model = str(cfg.model).strip().lower().replace("_", "-")
    if model in {"s-alpha", "salpha", "analytic"}:
        return SAlphaGeometry.from_config(cfg)
    if model in {"slab"}:
        return SlabGeometry.from_config(cfg)
    if model in {
        "imported-netcdf",
        "imported-nc",
        "imported-eik",
        "netcdf",
        "nc",
        "eik",
        "vmec-eik",
        "desc-eik",
    }:
        if cfg.geometry_file is None:
            raise ValueError(
                "geometry.geometry_file must be set for imported NetCDF/eik geometry"
            )
        return load_imported_geometry_netcdf(cfg.geometry_file)
    raise ValueError(
        "geometry.model must be one of "
        "{'s-alpha', 'slab', 'imported-netcdf', 'imported-eik', 'vmec-eik', 'desc-eik'}"
    )


def ensure_flux_tube_geometry_data(
    geom: FluxTubeGeometryLike,
    theta: jnp.ndarray,
) -> FluxTubeGeometryData:
    """Return sampled geometry data for analytic or pre-sampled inputs."""

    if isinstance(geom, FluxTubeGeometryData):
        try:
            geom._theta_matches(theta)
            return geom
        except ValueError as exc:
            theta_arr = jnp.asarray(theta)
            if geom.theta.shape[0] == theta_arr.shape[-1] + 1:
                trimmed = geom.trim_terminal_theta_point()
                trimmed._theta_matches(theta)
                return trimmed
            raise exc
    return sample_flux_tube_geometry(geom, theta)


def twist_shift_params(
    geom: FluxTubeGeometryLike,
    grid: GridConfig,
) -> tuple[int, float]:
    """Return `(jtwist, x0)` for twist-and-shift grid defaults."""

    y0 = float(grid.y0) if grid.y0 is not None else float(grid.Ly) / (2.0 * jnp.pi)
    if isinstance(geom, FluxTubeGeometryData):
        gds21_val = float(np.asarray(geom.gds21_profile[0]))
        gds22_val = float(np.asarray(geom.gds22_profile[0]))
        shat = float(geom.s_hat)
    else:
        if grid.ntheta is not None:
            if grid.zp is not None:
                zp = int(grid.zp)
            elif grid.nperiod is not None:
                zp = 2 * int(grid.nperiod) - 1
            else:
                zp = 1
            theta_min = -jnp.pi * float(zp)
        else:
            theta_min = float(grid.z_min)
        theta_min_f = float(theta_min)
        shat = float(geom.s_hat)
        if isinstance(geom, SAlphaGeometry):
            shear = shat * theta_min_f - float(geom.alpha) * math.sin(theta_min_f)
            gds21_val = -shat * shear
            gds22_val = shat * shat
        elif isinstance(geom, SlabGeometry):
            shear = shat * theta_min_f
            gds21_val = -shat * shear
            gds22_val = 1.0 if shat == 0.0 else shat * shat
        else:
            _gds2, gds21, gds22 = geom.metric_coeffs(
                np.asarray([theta_min_f], dtype=float)
            )
            gds21_val = float(np.asarray(gds21, dtype=float).reshape(-1)[0])
            gds22_arr = np.asarray(gds22, dtype=float)
            gds22_val = (
                float(gds22_arr.reshape(-1)[0])
                if gds22_arr.ndim > 0
                else float(gds22_arr)
            )
    twist_shift_geo_fac = (
        2.0 * shat * gds21_val / gds22_val if gds22_val != 0.0 else 0.0
    )
    if grid.jtwist is None:
        jtwist = int(round(twist_shift_geo_fac))
        if jtwist == 0:
            jtwist = 1
    else:
        jtwist = int(grid.jtwist)
        if jtwist == 0:
            jtwist = 1
    if twist_shift_geo_fac == 0.0:
        x0 = y0
    else:
        x0 = y0 * abs(jtwist) / abs(twist_shift_geo_fac)
    return jtwist, x0


def apply_geometry_grid_defaults(
    geom: FluxTubeGeometryLike,
    grid: GridConfig,
) -> GridConfig:
    """Apply imported-geometry grid defaults implied by the selected geometry."""

    grid_out = grid
    if isinstance(geom, FluxTubeGeometryData):
        theta = np.asarray(geom.theta, dtype=float)
        if theta.ndim != 1 or theta.size < 2:
            raise ValueError(
                "Imported geometry theta grid must be one-dimensional with at least two points"
            )
        if geom.theta_closed_interval:
            nz = int(theta.size - 1)
            z_min = float(theta[0])
            z_max = float(theta[-1])
        else:
            spacing = float(theta[1] - theta[0])
            nz = int(theta.size)
            z_min = float(theta[0])
            z_max = float(theta[-1] + spacing)
        grid_out = replace(
            grid_out,
            Nz=nz,
            z_min=z_min,
            z_max=z_max,
            ntheta=None,
            nperiod=None,
            zp=None,
        )
        if float(grid_out.kxfac) == 1.0:
            grid_out = replace(grid_out, kxfac=float(geom.kxfac))
    boundary = effective_boundary(
        str(grid_out.boundary).lower(),
        s_hat=float(getattr(geom, "s_hat", 0.0)),
        zero_shat=bool(getattr(geom, "zero_shat", False)),
    )
    if boundary != str(grid_out.boundary).lower():
        grid_out = replace(grid_out, boundary=boundary, jtwist=None)
    if boundary in {"linked", "fix aspect"} and not bool(grid_out.non_twist):
        jtwist, x0 = twist_shift_params(geom, grid_out)
        grid_out = replace(grid_out, Lx=2.0 * np.pi * x0, jtwist=jtwist)
    elif boundary == "periodic" and zero_shear_enabled(
        float(getattr(geom, "s_hat", 0.0)),
        zero_shat=bool(getattr(geom, "zero_shat", False)),
    ):
        # Zero-shear promotion switches the lane onto the periodic
        # grad-parallel operator, so any linked-FFT metadata must be cleared.
        grid_out = replace(grid_out, jtwist=None)
    return grid_out


apply_imported_geometry_grid_defaults = apply_geometry_grid_defaults
