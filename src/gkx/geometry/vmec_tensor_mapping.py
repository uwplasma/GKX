"""Thin GKX conversions for VMEX-owned toroidal, WOUT, and mirror mappings."""

from __future__ import annotations

import importlib
from typing import Any


def _import_vmex(module: str, message: str) -> Any:
    """Import an optional VMEX owner only when its adapter is called."""
    try:
        return importlib.import_module(module)
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(message) from exc


def _vmex_mapping(method: str, *args: Any, **selection: Any) -> dict[str, Any]:
    owner = _import_vmex("vmex.core.turbulence", "vmex is required for VMEC geometry")
    mapping = dict(getattr(owner, method)(*args, **selection))
    metadata = dict(mapping.pop("vmex"))
    mapping["vmex"] = {
        **metadata,
        "reference_length": metadata["L_ref"],
        "reference_b": metadata["B_ref"],
    }
    return mapping


def _geometry(mapping: dict[str, Any], source: str, validate_finite: bool) -> Any:
    from gkx.geometry.flux_tube_contract import flux_tube_geometry_from_mapping

    return flux_tube_geometry_from_mapping(
        mapping, source_model=source, validate_finite=validate_finite
    )


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
    """Return VMEX's differentiable equal-arc PEST field-line mapping."""
    return _vmex_mapping(
        "gk_fieldline_geometry",
        state,
        runtime,
        s_index=None if surface_index is None else int(surface_index),
        alpha=float(alpha),
        zeta0=float(zeta0),
        ntheta=int(ntheta),
        equal_arc=bool(equal_arc),
        arc_oversample=int(arc_oversample),
    )


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
    mapping = vmex_flux_tube_mapping_from_state(
        state, runtime, surface_index=surface_index, alpha=alpha, zeta0=zeta0,
        ntheta=ntheta, equal_arc=equal_arc, arc_oversample=arc_oversample,
    )
    return _geometry(mapping, "vmex:core.turbulence", validate_finite)


def from_vmex_wout(wout: Any, **geometry_kwargs: Any) -> Any:
    """Return generic GKX geometry from a VMEC-compatible WOUT via VMEX."""
    validate_finite = bool(geometry_kwargs.pop("validate_finite", True))
    surface_index = geometry_kwargs.pop("surface_index", None)
    mapping = _vmex_mapping(
        "gk_fieldline_geometry_from_wout",
        wout,
        s_index=None if surface_index is None else int(surface_index),
        **geometry_kwargs,
    )
    return _geometry(mapping, "vmex:core.turbulence:wout", validate_finite)


def from_vmex_mirror(
    state: Any,
    discretization: Any,
    axis: Any,
    *,
    validate_finite: bool = True,
    **geometry_kwargs: Any,
) -> Any:
    """Return generic GKX geometry for a periodic VMEX mirror field line."""
    mirror = _import_vmex(
        "vmex.mirror.turbulence", "vmex closed-mirror geometry support is required"
    )
    return _geometry(
        mirror.gk_closed_fieldline_geometry(
            state, discretization, axis, **geometry_kwargs
        ),
        "vmex:mirror.turbulence",
        validate_finite,
    )


__all__ = ["from_vmex", "from_vmex_mirror", "from_vmex_wout",
           "vmex_flux_tube_mapping_from_state"]
