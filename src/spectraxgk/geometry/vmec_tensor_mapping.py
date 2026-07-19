"""Direct VMEC (PEST field-line) flux-tube mapping bridge via ``vmex``.

The historical tensor route evaluated raw VMEC metric/``|B|`` tensors from the
retired ``vmec_jax`` internals and rebuilt the flux-tube contract locally.
The route is now a thin adapter over the vmex public turbulence seam
:func:`vmex.core.turbulence.gk_fieldline_geometry`, which emits the exact
in-memory mapping contract consumed by
:func:`spectraxgk.flux_tube_geometry_from_mapping` (GS2-style normalizations,
PEST field-line sampling, pure JAX, differentiable w.r.t. ``(state,
runtime)``).
"""

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


def vmec_jax_flux_tube_mapping_from_state(  # pragma: no cover
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
    """Build a solver-ready flux-tube mapping from a solved ``vmex`` state.

    ``state``/``runtime`` come from :func:`load_solved_vmex_case` (or any
    :class:`vmex.optimize.Equilibrium`).  The mapping is produced by
    :func:`vmex.core.turbulence.gk_fieldline_geometry`: one PEST field line
    ``theta* = alpha + iota (phi - zeta0)`` on full-mesh surface
    ``surface_index`` (vmex default, ~60 % of the radius, when ``None``),
    sampled on ``theta = linspace(-pi, pi, ntheta, endpoint=False)`` with the
    GS2-style normalizations ``L_ref`` = effective minor radius and
    ``B_ref = 2 |psi_edge| / L_ref**2``.  ``equal_arc=True`` (the SPECTRAX-GK
    solver contract) resamples the parallel coordinate so ``gradpar`` is
    exactly uniform.

    The returned dict satisfies :func:`flux_tube_geometry_from_mapping` and is
    differentiable with respect to the vmex spectral state.  The vmex parity
    diagnostics (``dp_drho``, ``gradpar_profile``, sampled PEST angles, ...)
    are passed through under the ``"vmec_jax"`` key expected by the existing
    report consumers, with ``reference_length``/``reference_b`` aliases for
    ``L_ref``/``B_ref``.
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
    mapping["vmec_jax"] = {
        **vmex_meta,
        "reference_length": vmex_meta["L_ref"],
        "reference_b": vmex_meta["B_ref"],
    }
    return mapping


__all__ = ["vmec_jax_flux_tube_mapping_from_state"]
