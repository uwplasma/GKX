"""Resolve a deck ``collision_operator`` name to a solver collision operator.

This factory lived in ``operators/linear/params.py``, which is the package's
highest fan-in module: 99 names are imported from it, resolving to a handful of
shared types. A factory that reaches forward into the collision tables does not
belong in a type module -- it was the only reason ``params`` imported upward at
all, and the import had to be deferred inside the function body to avoid a
cycle. Here it needs no deferral.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp

from gkx.operators.collision import CollisionOperator
from gkx.operators.linear.params import COLLISION_OPERATOR_NAMES


def collision_operator_from_config(
    name: str,
    *,
    density: jnp.ndarray,
    mass: jnp.ndarray,
    temperature: jnp.ndarray,
    nu: jnp.ndarray | float = 1.0,
    moments: int = 8,
) -> CollisionOperator | None:
    """Resolve a TOML ``collision_operator`` name to a solver collision operator.

    ``"none"`` and ``"lenard_bernstein"`` return ``None`` so the linear RHS
    keeps its built-in diagonal Lenard-Bernstein term (the solver re-enables
    ``collisions_contribution`` exactly when ``collision_operator is None``).
    ``"sugama"``, ``"improved_sugama"``, and ``"coulomb"`` build the dense
    drift-kinetic Hermite-Laguerre moment operator (Frei, Ernst & Ricci 2022)
    that replaces the diagonal term. ``"coulomb"`` is the full linearized
    Coulomb (Landau) operator of equations (C9a)--(C9f) and is validated for
    like-species collisions only. ``density``/``mass``/``temperature`` are the
    per-species normalizations (length ``n_species``).

    The assembled matrices carry only the dimensionless pair scaling
    ``n_b / (sqrt(m_a) T_a**1.5)``; the common collisionality prefactor is the
    caller's responsibility, so ``nu`` is applied here. It is the same ``nu``
    that sets the strength of the built-in Lenard-Bernstein term, which keeps
    every model on one collisionality axis.
    """

    from gkx.operators.linear.collisions import DriftKineticMomentCollisionOperator

    key = name.strip().lower()
    if key in ("none", "lenard_bernstein"):
        return None

    collisionality = jnp.asarray(nu)
    if collisionality.ndim > 1:
        raise ValueError("nu must be a scalar or a per-species vector")
    if collisionality.ndim == 1:
        # A per-species vector scales each target species' row block.
        collisionality = collisionality.reshape((-1, 1, 1, 1))

    if key == "sugama":
        operator = DriftKineticMomentCollisionOperator.from_species(
            density, mass, temperature
        )
    elif key == "improved_sugama":
        operator = DriftKineticMomentCollisionOperator.from_improved_species(
            density, mass, temperature
        )
    elif key == "coulomb":
        from gkx.operators.linear.collision_tables import (
            assemble_drift_kinetic_coulomb_matrix,
        )

        operator = DriftKineticMomentCollisionOperator(
            assemble_drift_kinetic_coulomb_matrix(density, mass, temperature)
        )
    elif key == "coulomb_finite_kperp":
        from gkx.operators.linear.collision_tables import (
            build_finite_wavelength_coulomb_operator,
        )

        finite = build_finite_wavelength_coulomb_operator(
            density, mass, temperature, moments
        )
        scale = (
            jnp.reshape(jnp.asarray(nu), ())
            if jnp.asarray(nu).ndim == 0
            else (jnp.asarray(nu).reshape((-1, 1)))
        )
        return dataclasses.replace(finite, pair_frequency=finite.pair_frequency * scale)
    else:
        raise ValueError(
            f"collision_operator must be one of {COLLISION_OPERATOR_NAMES}"
        )
    return DriftKineticMomentCollisionOperator(operator.matrix * collisionality)
