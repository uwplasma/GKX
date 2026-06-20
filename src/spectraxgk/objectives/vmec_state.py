"""VMEC/Boozer state coefficient helpers for solver-objective gates."""

from __future__ import annotations

from dataclasses import replace as dc_replace
from typing import Any

import jax.numpy as jnp


VMEC_BOOZER_STATE_PARAMETER_NAMES = ("Rcos_mid_surface_m1",)
VMEC_BOOZER_STATE_PARAMETER_FAMILIES = ("Rcos", "Rsin", "Zcos", "Zsin", "Lcos", "Lsin")


def _vmec_boozer_state_array(state: Any, parameter_family: str) -> jnp.ndarray:
    family = str(parameter_family)
    if family not in VMEC_BOOZER_STATE_PARAMETER_FAMILIES:
        raise ValueError(
            "parameter_family must be one of "
            f"{', '.join(VMEC_BOOZER_STATE_PARAMETER_FAMILIES)}"
        )
    if not hasattr(state, family):
        raise RuntimeError(f"vmec_jax state does not expose {family}")
    array = jnp.asarray(getattr(state, family))
    if array.ndim != 2 or int(array.shape[1]) < 2:
        raise RuntimeError(
            f"vmec_jax state {family} array must expose at least one non-axisymmetric mode"
        )
    return array


def _replace_vmec_boozer_state_coefficient(
    state: Any,
    parameter_family: str,
    base_array: jnp.ndarray,
    radial_index: int,
    mode_index: int,
    delta: Any,
) -> Any:
    return dc_replace(
        state,
        **{
            str(parameter_family): base_array.at[
                int(radial_index),
                int(mode_index),
            ].add(delta)
        },
    )


def _vmec_boozer_state_parameter_name(
    parameter_family: str,
    radial_index: int,
    mode_index: int,
    *,
    default_mid_surface: int,
) -> str:
    family = str(parameter_family)
    if int(radial_index) == int(default_mid_surface):
        return f"{family}_mid_surface_m{int(mode_index)}"
    return f"{family}_r{int(radial_index)}_m{int(mode_index)}"


__all__ = [
    "VMEC_BOOZER_STATE_PARAMETER_FAMILIES",
    "VMEC_BOOZER_STATE_PARAMETER_NAMES",
    "_replace_vmec_boozer_state_coefficient",
    "_vmec_boozer_state_array",
    "_vmec_boozer_state_parameter_name",
]
