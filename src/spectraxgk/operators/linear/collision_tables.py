"""Tabulated collision matrices and differentiable wavelength interpolation."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.core.extension_points import CollisionContext
from spectraxgk.operators.linear.collisions import (
    apply_multispecies_collision_moment_matrix,
)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class TabulatedMultispeciesCollisionOperator:
    """Finite-wavelength target/source collision matrices on a kperp grid.

    The table contains fully assembled collision-frequency-weighted blocks with
    shape ``(target, source, kperp, moment, moment)``. The runtime derives each
    target species' normalized ``kperp`` from ``sqrt(cache.b)`` and keeps the
    interpolation and matrix application inside JAX.
    """

    kperp_grid: jnp.ndarray
    matrices: jnp.ndarray

    def apply(self, context: CollisionContext) -> jnp.ndarray:
        """Interpolate and apply the table to the post-field Hamiltonian."""

        table = jnp.asarray(self.matrices)
        if table.ndim != 5:
            raise ValueError(
                "tabulated multispecies collision matrices must have target, "
                "source, kperp, and two moment axes"
            )
        kperp = jnp.sqrt(jnp.maximum(jnp.asarray(context.cache.b), 0.0))
        matrix = interpolate_collision_moment_matrix(self.kperp_grid, table, kperp)
        return apply_multispecies_collision_moment_matrix(context.hamiltonian, matrix)

    def tree_flatten(self):
        return (self.kperp_grid, self.matrices), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


def interpolate_collision_moment_matrix(
    kperp_grid: jnp.ndarray,
    matrices: jnp.ndarray,
    kperp: jnp.ndarray,
) -> jnp.ndarray:
    """Interpolate collision matrices onto a scalar or spatial ``kperp`` field.

    ``matrices`` may contain one shared table, one table per species, or one
    table per ordered target/source species pair. Their respective shapes are
    ``(kperp, modes, modes)``, ``(species, kperp, modes, modes)``, and
    ``(target, source, kperp, modes, modes)``. Values outside the tabulated
    interval use the nearest endpoint. The coefficient grid is validated on
    the host; interpolation and its derivative with respect to ``kperp``
    remain in JAX.
    """

    grid = jnp.asarray(kperp_grid)
    table = jnp.asarray(matrices)
    target = jnp.asarray(kperp, dtype=jnp.result_type(grid, table))
    if grid.ndim != 1 or int(grid.size) < 2:
        raise ValueError(
            "collision kperp grid must be one-dimensional with at least two points"
        )
    if table.ndim not in {3, 4, 5}:
        raise ValueError(
            "collision table must have shape (kperp, modes, modes) or "
            "(species, kperp, modes, modes), optionally with separate "
            "target/source species axes"
        )
    grid_axis = table.ndim - 3
    if int(table.shape[grid_axis]) != int(grid.size):
        raise ValueError("collision table kperp axis must match the coefficient grid")
    if int(table.shape[-1]) != int(table.shape[-2]):
        raise ValueError("collision table matrices must be square")
    if not isinstance(grid, jax.core.Tracer):
        host_grid = np.asarray(grid)
        if not np.all(np.isfinite(host_grid)) or not np.all(np.diff(host_grid) > 0.0):
            raise ValueError(
                "collision kperp grid must be finite and strictly increasing"
            )

    def interpolate_one(species_table: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
        clipped = jnp.clip(values, grid[0], grid[-1])
        left = jnp.clip(
            jnp.searchsorted(grid, clipped, side="right") - 1, 0, grid.size - 2
        )
        fraction = (clipped - grid[left]) / (grid[left + 1] - grid[left])
        interpolated = species_table[left] + fraction[..., None, None] * (
            species_table[left + 1] - species_table[left]
        )
        return jnp.moveaxis(interpolated, (-2, -1), (0, 1))

    if table.ndim == 3:
        return interpolate_one(table, target)
    species_count = int(table.shape[0])
    if table.ndim == 5:
        if int(table.shape[1]) != species_count:
            raise ValueError(
                "multispecies collision table must have equal target/source axes"
            )

        def interpolate_target_pairs(
            pair_tables: jnp.ndarray, values: jnp.ndarray
        ) -> jnp.ndarray:
            return jax.vmap(lambda pair_table: interpolate_one(pair_table, values))(
                pair_tables
            )

        if target.ndim == 0:
            return jax.vmap(lambda pairs: interpolate_target_pairs(pairs, target))(
                table
            )
        if int(target.shape[0]) != species_count:
            raise ValueError(
                "multispecies collision table requires scalar kperp or a "
                "target-species-leading kperp field"
            )
        return jax.vmap(interpolate_target_pairs)(table, target)
    if target.ndim == 0:
        return jax.vmap(lambda species_table: interpolate_one(species_table, target))(
            table
        )
    if int(target.shape[0]) != species_count:
        raise ValueError(
            "species collision table requires scalar kperp or a species-leading kperp field"
        )
    return jax.vmap(interpolate_one)(table, target)
