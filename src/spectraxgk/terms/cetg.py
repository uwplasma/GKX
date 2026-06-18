"""Compatibility facade for the collisional-slab ETG reduced model."""

from __future__ import annotations

from spectraxgk.terms.reduced.cetg_integrator import (
    _cetg_diag_weight,
    _cetg_flux_weight,
    _compute_cetg_diag,
    integrate_cetg_explicit_diagnostics_state,
)
from spectraxgk.terms.reduced.cetg_model import (
    CETGModelParams,
    build_cetg_model_params,
    validate_cetg_runtime_config,
)
from spectraxgk.terms.reduced.cetg_rhs import (
    _cetg_linear_omega_max,
    _cetg_linear_rhs,
    _cetg_nonlinear_omega_components,
    _cetg_nonlinear_rhs,
    cetg_fields,
    cetg_rhs,
)
from spectraxgk.terms.reduced.cetg_state import (
    _apply_kz_filter,
    _dz2,
    _from_internal_state,
    _kz_grid,
    _project_state,
    _to_internal_state,
    _xy_mask,
)

__all__ = [
    "CETGModelParams",
    "_apply_kz_filter",
    "_cetg_diag_weight",
    "_cetg_flux_weight",
    "_cetg_linear_omega_max",
    "_cetg_linear_rhs",
    "_cetg_nonlinear_omega_components",
    "_cetg_nonlinear_rhs",
    "_compute_cetg_diag",
    "_dz2",
    "_from_internal_state",
    "_kz_grid",
    "_project_state",
    "_to_internal_state",
    "_xy_mask",
    "build_cetg_model_params",
    "cetg_fields",
    "cetg_rhs",
    "integrate_cetg_explicit_diagnostics_state",
    "validate_cetg_runtime_config",
]
