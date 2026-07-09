"""Stable facade for the internal Miller imported-geometry backend."""

from __future__ import annotations

from spectraxgk.geometry_backends.miller_core import (
    MillerCoreParams,
    build_collocation_surfaces,
    compute_equal_arc_theta,
    compute_primary_gradients,
    compute_straight_field_theta,
    rebuild_straight_theta_state,
)
from spectraxgk.geometry_backends.miller_numerics import (
    _safe_denom,
    cumulative_trapezoid,
    derm,
    dermv,
    nperiod_data_extend,
    reflect_n_append,
    to_ballooning,
)
from spectraxgk.geometry_backends.miller_pipeline import (
    _request_attr,
    generate_miller_eik_internal,
    internal_miller_backend_available,
    write_miller_eik_netcdf,
)
from spectraxgk.geometry_backends.miller_profiles import assemble_miller_profiles

__all__ = [
    "MillerCoreParams",
    "_request_attr",
    "_safe_denom",
    "assemble_miller_profiles",
    "build_collocation_surfaces",
    "compute_equal_arc_theta",
    "compute_primary_gradients",
    "compute_straight_field_theta",
    "cumulative_trapezoid",
    "derm",
    "dermv",
    "generate_miller_eik_internal",
    "internal_miller_backend_available",
    "nperiod_data_extend",
    "rebuild_straight_theta_state",
    "reflect_n_append",
    "to_ballooning",
    "write_miller_eik_netcdf",
]
