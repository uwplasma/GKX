"""vmex state controls and parity metrics shared by the VMEC campaign scripts."""

from __future__ import annotations

from dataclasses import dataclass, replace as dc_replace
from typing import Any

import jax.numpy as jnp
import numpy as np

from gkx.geometry.vmec_boozer_core import (
    load_solved_vmex_case,
    resolve_vmex_case_input_path,
)


#: Provenance marker: vmex equilibria are solved in memory, without a wout file.
VMEC_STATE_IN_MEMORY_WOUT_PATH = "in-memory:vmex.optimize.solve_equilibrium"


@dataclass(frozen=True)
class _VMECStateContext:
    """Solved vmex example state plus coefficient arrays used by AD gates.

    ``base_Rcos``/``base_Zsin`` mirror the vmex ``R_cos``/``Z_sin`` spectral
    tables; ``wout_path`` is the in-memory provenance marker because vmex
    solves the equilibrium directly instead of reading a wout file.
    """

    input_path: Any
    wout_path: Any
    inp: Any
    runtime: Any
    wout: Any
    state: Any
    base_Rcos: jnp.ndarray
    base_Zsin: jnp.ndarray


def _load_vmec_state_context(case_name: str) -> _VMECStateContext:
    """Solve a bundled vmex example and expose differentiable state arrays."""

    input_path = resolve_vmex_case_input_path(str(case_name))
    inp, state, runtime, wout = load_solved_vmex_case(str(case_name))
    base_Rcos = jnp.asarray(state.R_cos)
    base_Zsin = jnp.asarray(state.Z_sin)
    if base_Rcos.ndim != 2 or base_Zsin.ndim != 2:
        raise RuntimeError("vmex state R_cos/Z_sin arrays must be two-dimensional")
    return _VMECStateContext(
        input_path=input_path,
        wout_path=VMEC_STATE_IN_MEMORY_WOUT_PATH,
        inp=inp,
        runtime=runtime,
        wout=wout,
        state=state,
        base_Rcos=base_Rcos,
        base_Zsin=base_Zsin,
    )


def _resolve_vmec_state_indices(
    base_Rcos: jnp.ndarray,
    *,
    radial_index: int | None,
    mode_index: int,
    surface_index: int | None,
    surface_grid: str,
) -> tuple[int, int, int]:
    """Resolve coefficient and surface indices for VMEC-state sensitivity gates."""

    ns_full = int(base_Rcos.shape[0])
    ridx = ns_full // 2 if radial_index is None else int(radial_index)
    midx = int(mode_index)
    if not (0 <= ridx < ns_full):
        raise ValueError("radial_index is outside the VMEC state radial grid")
    if not (0 <= midx < int(base_Rcos.shape[1])):
        raise ValueError("mode_index is outside the VMEC state mode table")

    if surface_grid == "half_mesh":
        default_sidx = max(0, min(ridx - 1, ns_full - 2))
        surface_count = ns_full - 1
        error = "surface_index is outside the VMEC half-mesh Boozer surface grid"
    elif surface_grid == "field_line":
        default_sidx = max(1, min(ridx, ns_full - 2))
        surface_count = ns_full
        error = "surface_index is outside the VMEC metric radial grid"
    elif surface_grid == "metric":
        default_sidx = max(0, min(ridx - 1, ns_full - 1))
        surface_count = ns_full
        error = "surface_index is outside the VMEC metric radial grid"
    else:
        raise ValueError(f"unknown VMEC surface grid {surface_grid!r}")

    sidx = default_sidx if surface_index is None else int(surface_index)
    if not (0 <= sidx < surface_count):
        raise ValueError(error)
    return int(ridx), int(midx), int(sidx)


def _perturb_vmec_state(
    ctx: _VMECStateContext,
    x: jnp.ndarray,
    *,
    radial_index: int,
    mode_index: int,
) -> Any:
    """Return a VMEC state with two Fourier controls perturbed by ``x``."""

    return dc_replace(
        ctx.state,
        R_cos=ctx.base_Rcos.at[radial_index, mode_index].add(x[0]),
        Z_sin=ctx.base_Zsin.at[radial_index, mode_index].add(x[1]),
    )


def _length_two_params(params: jnp.ndarray | None, default: float) -> jnp.ndarray:
    """Normalize optional VMEC control perturbations to a length-two vector."""

    p = jnp.asarray([default, default] if params is None else params, dtype=jnp.float64)
    if p.ndim != 1 or int(p.shape[0]) != 2:
        raise ValueError("params must be a length-2 vector")
    return p


def _array_parity_metrics(
    candidate: Any, reference: Any, *, floor: float = 1.0e-12
) -> dict[str, object]:
    cand = np.asarray(candidate, dtype=float)
    ref = np.asarray(reference, dtype=float)
    metrics: dict[str, object] = {
        "candidate_shape": [int(v) for v in cand.shape],
        "reference_shape": [int(v) for v in ref.shape],
        "shape_match": bool(cand.shape == ref.shape),
    }
    if cand.shape != ref.shape:
        return metrics
    diff = cand - ref
    ref_scale = max(float(np.nanmax(np.abs(ref))) if ref.size else 0.0, float(floor))
    local_scale = np.maximum(np.abs(ref), float(floor))
    metrics.update(
        {
            "max_abs": float(np.nanmax(np.abs(diff))) if diff.size else 0.0,
            "rms_abs": float(np.sqrt(np.nanmean(diff * diff))) if diff.size else 0.0,
            "max_rel_pointwise": float(np.nanmax(np.abs(diff) / local_scale))
            if diff.size
            else 0.0,
            "rms_rel_pointwise": float(np.sqrt(np.nanmean((diff / local_scale) ** 2)))
            if diff.size
            else 0.0,
            "reference_scale": ref_scale,
            "normalized_max_abs": float(np.nanmax(np.abs(diff)) / ref_scale)
            if diff.size
            else 0.0,
            "candidate_min": float(np.nanmin(cand)) if cand.size else 0.0,
            "candidate_max": float(np.nanmax(cand)) if cand.size else 0.0,
            "reference_min": float(np.nanmin(ref)) if ref.size else 0.0,
            "reference_max": float(np.nanmax(ref)) if ref.size else 0.0,
        }
    )
    return metrics


def _scalar_parity_metrics(
    candidate: Any, reference: Any, *, floor: float = 1.0e-12
) -> dict[str, float]:
    cand = float(np.asarray(candidate))
    ref = float(np.asarray(reference))
    diff = cand - ref
    scale = max(abs(ref), float(floor))
    return {
        "candidate": cand,
        "reference": ref,
        "abs": abs(diff),
        "rel": abs(diff) / scale,
    }
