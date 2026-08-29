"""Thin GKX conversions for VMEX-owned toroidal and closed-mirror mappings."""

from __future__ import annotations

import importlib
from typing import Any


def _import_vmex_turbulence() -> Any:
    """Import the vmex GK field-line geometry seam."""

    try:
        return importlib.import_module("vmex.core.turbulence")
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "vmex is required for the direct VMEC flux-tube mapping"
        ) from exc


def vmex_flux_tube_mapping_from_state(  # pragma: no cover
    state: Any,
    runtime: Any,
    *,
    surface_index: int | None = None,
    alpha: float = 0.0,
    zeta0: float = 0.0,
    ntheta: int = 32,
    equal_arc: bool = True,
    arc_oversample: int = 4,
) -> dict[str, Any]:
    """Build VMEX's equal-arc PEST mapping with GS2/GX normalizations.

    VMEX owns all arrays and differentiability.  Its diagnostics pass through
    under ``"vmex"`` with the two historical reference aliases GKX reports use.
    """

    turbulence_mod = _import_vmex_turbulence()
    mapping = dict(
        turbulence_mod.gk_fieldline_geometry(
            state,
            runtime,
            s_index=None if surface_index is None else int(surface_index),
            alpha=float(alpha),
            zeta0=float(zeta0),
            ntheta=int(ntheta),
            equal_arc=bool(equal_arc),
            arc_oversample=int(arc_oversample),
        )
    )
    vmex_meta = dict(mapping.pop("vmex"))
    mapping["vmex"] = {
        **vmex_meta,
        "reference_length": vmex_meta["L_ref"],
        "reference_b": vmex_meta["B_ref"],
    }
    return mapping


def from_vmex(
    state: Any,
    runtime: Any,
    *,
    surface_index: int | None = None,
    alpha: float = 0.0,
    zeta0: float = 0.0,
    ntheta: int = 32,
    equal_arc: bool = True,
    arc_oversample: int = 4,
    validate_finite: bool = True,
) -> Any:
    """Return generic GKX geometry for a solved VMEX toroidal state."""

    from gkx.geometry.flux_tube_contract import flux_tube_geometry_from_mapping

    mapping = vmex_flux_tube_mapping_from_state(
        state,
        runtime,
        surface_index=surface_index,
        alpha=alpha,
        zeta0=zeta0,
        ntheta=ntheta,
        equal_arc=equal_arc,
        arc_oversample=arc_oversample,
    )
    return flux_tube_geometry_from_mapping(
        mapping,
        source_model="vmex:core.turbulence",
        validate_finite=validate_finite,
    )


def from_vmex_mirror(
    state: Any,
    discretization: Any,
    axis: Any,
    *,
    validate_finite: bool = True,
    **geometry_kwargs: Any,
) -> Any:
    """Return generic GKX geometry for a periodic VMEX mirror field line."""

    try:
        mirror = importlib.import_module("vmex.mirror.turbulence")
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("vmex closed-mirror geometry support is required") from exc
    from gkx.geometry.flux_tube_contract import flux_tube_geometry_from_mapping

    return flux_tube_geometry_from_mapping(
        mirror.gk_closed_fieldline_geometry(
            state, discretization, axis, **geometry_kwargs
        ),
        source_model="vmex:mirror.turbulence",
        validate_finite=validate_finite,
    )


__all__ = ["from_vmex", "from_vmex_mirror", "vmex_flux_tube_mapping_from_state"]
