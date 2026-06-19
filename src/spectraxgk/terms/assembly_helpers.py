"""Shared helper policies for gyrokinetic RHS assembly."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.terms.config import FieldState, TermConfig
from spectraxgk.terms.linear_dissipation import collisions_contribution


def _apply_external_phi_source(
    fields: FieldState,
    *,
    external_phi: jnp.ndarray | float | None,
) -> FieldState:
    """Apply a external electrostatic source after the field solve."""

    if external_phi is None:
        return fields
    phi_shift = jnp.asarray(external_phi, dtype=fields.phi.dtype)
    if phi_shift.ndim > fields.phi.ndim:
        raise ValueError("external_phi must be broadcastable to the solved phi field")
    return FieldState(phi=fields.phi + phi_shift, apar=fields.apar, bpar=fields.bpar)


def _is_static_zero(value: object) -> bool:
    """Return true when a Python/JAX value is known to be exactly zero at trace time."""

    arr = jnp.asarray(value)
    if isinstance(arr, jax.core.Tracer):
        return False
    return bool(np.all(np.asarray(arr) == 0.0))


def _rhs_field_views(
    fields: FieldState,
    terms: TermConfig,
    *,
    force_electrostatic_fields: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None]:
    """Return zero-filled RHS fields and optional Hamiltonian fields.

    Streaming and diamagnetic terms expect array-valued electromagnetic fields.
    ``build_H`` already has an electrostatic path when these fields are ``None``;
    keeping disabled fields as ``None`` there avoids compiling zero-valued
    electromagnetic Hamiltonian branches in electrostatic nonlinear runs.
    """

    apar = fields.apar if fields.apar is not None else jnp.zeros_like(fields.phi)
    bpar = fields.bpar if fields.bpar is not None else jnp.zeros_like(fields.phi)
    h_apar = (
        None
        if force_electrostatic_fields
        or fields.apar is None
        or _is_static_zero(terms.apar)
        else fields.apar
    )
    h_bpar = (
        None
        if force_electrostatic_fields
        or fields.bpar is None
        or _is_static_zero(terms.bpar)
        else fields.bpar
    )
    return apar, bpar, h_apar, h_bpar


def _collision_contribution_or_zero(
    H: jnp.ndarray,
    *,
    G: jnp.ndarray,
    Jl: jnp.ndarray,
    JlB: jnp.ndarray,
    b: jnp.ndarray,
    nu: jnp.ndarray,
    collision_lam: jnp.ndarray,
    lb_lam: jnp.ndarray,
    weight: jnp.ndarray,
) -> jnp.ndarray:
    """Skip the collision operator when the configured operator is exactly zero."""

    real_dtype = jnp.real(G).dtype

    def _is_static_zero(value: jnp.ndarray) -> bool:
        arr = jnp.asarray(value, dtype=real_dtype)
        if isinstance(arr, jax.core.Tracer):
            return False
        return bool(np.all(np.asarray(arr) == 0.0))

    if _is_static_zero(weight):
        return jnp.zeros_like(H)

    no_preexpanded_operator = collision_lam.size == 0
    if no_preexpanded_operator and _is_static_zero(nu):
        return jnp.zeros_like(H)

    zero_nu_operator = jnp.logical_and(no_preexpanded_operator, jnp.all(nu == 0.0))
    zero_weight = jnp.all(weight == 0.0)
    skip = jnp.logical_or(zero_weight, zero_nu_operator)

    return jax.lax.cond(
        skip,
        lambda _: jnp.zeros_like(H),
        lambda _: collisions_contribution(
            H,
            G=G,
            Jl=Jl,
            JlB=JlB,
            b=b,
            nu=nu,
            collision_lam=collision_lam,
            lb_lam=lb_lam,
            weight=weight,
        ),
        operand=None,
    )

__all__ = [
    "_apply_external_phi_source",
    "_collision_contribution_or_zero",
    "_is_static_zero",
    "_rhs_field_views",
]
