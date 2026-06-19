"""Shared VMEC-JAX state control helpers for differentiable geometry gates."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, replace as dc_replace
from typing import Any

import jax.numpy as jnp


@dataclass(frozen=True)
class _VMECStateContext:
    """Loaded VMEC-JAX example state plus coefficient arrays used by AD gates."""

    input_path: Any
    wout_path: Any
    cfg: Any
    indata: Any
    static: Any
    wout: Any
    state: Any
    base_Rcos: jnp.ndarray
    base_Zsin: jnp.ndarray


def _load_vmec_state_context(case_name: str) -> _VMECStateContext:
    """Load a bundled VMEC-JAX example and expose differentiable state arrays."""

    driver = importlib.import_module("vmec_jax.driver")
    config_mod = importlib.import_module("vmec_jax.config")
    static_mod = importlib.import_module("vmec_jax.static")
    wout_mod = importlib.import_module("vmec_jax.wout")

    input_path, wout_path = driver.example_paths(str(case_name))
    if wout_path is None:
        raise RuntimeError(f"vmec_jax example {case_name!r} has no bundled wout reference")

    cfg, indata = config_mod.load_config(str(input_path))
    static = static_mod.build_static(cfg)
    wout = wout_mod.read_wout(wout_path)
    state = wout_mod.state_from_wout(wout)
    base_Rcos = jnp.asarray(state.Rcos)
    base_Zsin = jnp.asarray(state.Zsin)
    if base_Rcos.ndim != 2 or base_Zsin.ndim != 2:
        raise RuntimeError("vmec_jax state Rcos/Zsin arrays must be two-dimensional")
    return _VMECStateContext(
        input_path=input_path,
        wout_path=wout_path,
        cfg=cfg,
        indata=indata,
        static=static,
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
        Rcos=ctx.base_Rcos.at[radial_index, mode_index].add(x[0]),
        Zsin=ctx.base_Zsin.at[radial_index, mode_index].add(x[1]),
    )


def _length_two_params(params: jnp.ndarray | None, default: float) -> jnp.ndarray:
    """Normalize optional VMEC control perturbations to a length-two vector."""

    p = jnp.asarray([default, default] if params is None else params, dtype=jnp.float64)
    if p.ndim != 1 or int(p.shape[0]) != 2:
        raise ValueError("params must be a length-2 vector")
    return p


__all__ = [
    "_VMECStateContext",
    "_length_two_params",
    "_load_vmec_state_context",
    "_perturb_vmec_state",
    "_resolve_vmec_state_indices",
]
