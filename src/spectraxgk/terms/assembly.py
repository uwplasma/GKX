"""Public RHS assembly facade for term-wise gyrokinetic evolution."""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp

from spectraxgk.core.grid import SpectralGrid
from spectraxgk.geometry import FluxTubeGeometryLike
from spectraxgk.operators.linear.cache_model import LinearCache
from spectraxgk.operators.linear.cache_builder import build_linear_cache
from spectraxgk.operators.linear.params import LinearParams
from spectraxgk.terms.assembly_core import (
    assemble_rhs_cached,
    assemble_rhs_cached_electrostatic_jit,
    assemble_rhs_cached_jit,
)
from spectraxgk.terms.assembly_diagnostics import assemble_rhs_terms_cached
from spectraxgk.terms.assembly_fields import compute_fields_cached
from spectraxgk.terms.assembly_helpers import (
    _apply_external_phi_source,
    _collision_contribution_or_zero,
    _is_static_zero,
    _rhs_field_views,
)
from spectraxgk.terms.config import FieldState, TermConfig

def assemble_rhs(
    G: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    *,
    Nl: int,
    Nm: int,
    terms: TermConfig | None = None,
    cache: LinearCache | None = None,
    dt: jnp.ndarray | float | None = None,
    external_phi: jnp.ndarray | float | None = None,
) -> Tuple[jnp.ndarray, FieldState]:
    """Assemble the RHS from term-wise modules."""

    cache = cache or build_linear_cache(grid, geom, params, Nl, Nm)
    return assemble_rhs_cached(
        G, cache, params, terms=terms, dt=dt, external_phi=external_phi
    )

__all__ = [
    "_apply_external_phi_source",
    "_collision_contribution_or_zero",
    "_is_static_zero",
    "_rhs_field_views",
    "assemble_rhs",
    "assemble_rhs_cached",
    "assemble_rhs_cached_electrostatic_jit",
    "assemble_rhs_cached_jit",
    "assemble_rhs_terms_cached",
    "compute_fields_cached",
]
