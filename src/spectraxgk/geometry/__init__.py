"""Analytic, slab, imported, and sampled flux-tube geometry contracts."""

from __future__ import annotations

from spectraxgk.config import GeometryConfig, GridConfig
from spectraxgk.geometry.analytic import (
    SAlphaGeometry,
    SlabGeometry,
    ZERO_SHAT_THRESHOLD,
    effective_boundary,
    zero_shear_enabled,
)
from spectraxgk.geometry.core import (
    FluxTubeGeometryLike,
    apply_geometry_grid_defaults,
    apply_imported_geometry_grid_defaults,
    build_flux_tube_geometry,
    ensure_flux_tube_geometry_data,
    twist_shift_params,
)
from spectraxgk.geometry.flux_tube import (
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
