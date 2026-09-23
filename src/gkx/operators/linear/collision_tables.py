"""Tabulated collision matrices and differentiable wavelength interpolation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import io
import json
from importlib import resources
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from gkx.operators.collision import CollisionContext
from gkx.operators.linear.collisions import (
    EqualSpeciesFiniteWavelengthCoulombOperator,
    apply_multispecies_collision_moment_matrix,
    interpolate_collision_diagonal_table,
    load_collision_moment_matrix,
)


_FINITE_WAVELENGTH_STEM = "finite_wavelength_coulomb"
# Shipped resolutions, keyed by Hermite-Laguerre moment count (P+1)*(J+1). The
# default eight-moment table matches the drift-kinetic tables so the two
# operators can be compared at equal resolution; the larger ones exist because
# published convergence studies need considerably more.
FINITE_WAVELENGTH_MOMENT_COUNTS: tuple[int, ...] = (8, 18)
_FINITE_WAVELENGTH_BLOCKS = (
    "test_matrix",
    "field_matrix",
    "test_phi1",
    "field_phi1",
    "test_phi2",
    "field_phi2",
)


@lru_cache(maxsize=len(FINITE_WAVELENGTH_MOMENT_COUNTS))
def _finite_wavelength_coulomb_bundle(
    moments: int = 8,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load and provenance-check one shipped finite-wavelength Coulomb table.

    ``moments`` is the Hermite-Laguerre moment count ``(P+1)*(J+1)``, which must
    equal the run's ``Nl*Nm``.
    """

    if moments not in FINITE_WAVELENGTH_MOMENT_COUNTS:
        raise ValueError(
            f"no shipped finite-wavelength Coulomb table with {moments} moments; "
            f"available: {sorted(FINITE_WAVELENGTH_MOMENT_COUNTS)}. Generate more "
            "with tools/artifacts/build_finite_wavelength_coulomb_data.py"
        )
    suffix = "" if moments == 8 else f"_{moments}"
    data_root = resources.files("gkx").joinpath("data")
    payload = data_root.joinpath(f"{_FINITE_WAVELENGTH_STEM}{suffix}.npz").read_bytes()
    metadata = json.loads(
        data_root.joinpath(f"{_FINITE_WAVELENGTH_STEM}{suffix}.json").read_text(
            encoding="utf-8"
        )
    )
    if hashlib.sha256(payload).hexdigest() != metadata.get("sha256"):
        raise ValueError(
            "finite-wavelength Coulomb coefficient checksum does not match metadata"
        )
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    missing = set(_FINITE_WAVELENGTH_BLOCKS + ("bessel_argument_grid",)) - set(arrays)
    if missing:
        raise ValueError(
            f"finite-wavelength Coulomb tables are missing {sorted(missing)}"
        )
    return arrays, metadata


def finite_wavelength_coulomb_metadata(moments: int = 8) -> dict[str, Any]:
    """Return the provenance metadata of one shipped Coulomb table."""

    return dict(_finite_wavelength_coulomb_bundle(moments)[1])


def build_finite_wavelength_coulomb_operator(
    density: jnp.ndarray,
    mass: jnp.ndarray,
    temperature: jnp.ndarray,
    moments: int = 8,
) -> EqualSpeciesFiniteWavelengthCoulombOperator:
    r"""Build the gyrokinetic (finite-Larmor) Coulomb operator for one species.

    This is the full linearized Coulomb operator retaining finite perpendicular
    wavelength, from Frei, Ball, Hoffmann, Jorge, Ricci & Stenger (2021),
    arXiv:2104.11480, equations (3.47)--(3.50). The shipped tables are indexed
    by the Bessel argument :math:`B=k_\perp v_{\mathrm{th}}/\Omega`; the runtime
    interpolates them at :math:`B=\sqrt{2b}` from the cached ``b``, so the same
    operator covers every perpendicular wavenumber in the simulation.

    As :math:`k_\perp\to0` the tables reduce to the drift-kinetic Coulomb
    operator, which is enforced by a physics gate.

    ``moments`` selects the shipped Hermite-Laguerre resolution and must equal
    the run's ``Nl*Nm``; see ``FINITE_WAVELENGTH_MOMENT_COUNTS``.

    The shipped tables are generated at ``mass_ratio = temperature_ratio = 1``,
    so they are valid for like-species collisions; a multi-species request is
    refused rather than silently extrapolated.
    """

    density_s = jnp.atleast_1d(jnp.asarray(density))
    mass_s = jnp.atleast_1d(jnp.asarray(mass, dtype=jnp.result_type(density_s, float)))
    temperature_s = jnp.atleast_1d(
        jnp.asarray(temperature, dtype=jnp.result_type(density_s, mass_s, float))
    )
    if mass_s.shape != density_s.shape or temperature_s.shape != density_s.shape:
        raise ValueError("density, mass, and temperature must have equal length")
    for value, name in (
        (density_s, "density"),
        (mass_s, "mass"),
        (temperature_s, "temperature"),
    ):
        if not isinstance(value, jax.core.Tracer) and np.any(np.asarray(value) <= 0.0):
            raise ValueError(f"{name} must be positive")

    species_count = int(density_s.size)
    if species_count != 1:
        raise ValueError(
            "the shipped finite-wavelength Coulomb tables are generated for "
            f"like-species collisions, but {species_count} species were given; "
            "use collision_operator='sugama' or 'improved_sugama' for "
            "multispecies runs"
        )

    arrays, _ = _finite_wavelength_coulomb_bundle(moments)
    dtype = jnp.result_type(density_s, mass_s, temperature_s, float)
    # Same dimensionless normalization as the drift-kinetic assemblers.
    frequency = density_s[0] / (jnp.sqrt(mass_s[0]) * temperature_s[0] ** 1.5)
    # The stored block names are the generator's; the operator names its two
    # matrix fields *_table.
    field_names = {
        "test_matrix": "test_table",
        "field_matrix": "field_table",
        "test_phi1": "test_phi1",
        "field_phi1": "field_phi1",
        "test_phi2": "test_phi2",
        "field_phi2": "field_phi2",
    }
    return EqualSpeciesFiniteWavelengthCoulombOperator(
        bessel_argument_grid=jnp.asarray(arrays["bessel_argument_grid"], dtype=dtype),
        pair_frequency=jnp.reshape(frequency, (1, 1)).astype(dtype),
        **{
            field: jnp.asarray(arrays[stored], dtype=dtype)
            for stored, field in field_names.items()
        },
    )


def assemble_drift_kinetic_coulomb_matrix(
    density: jnp.ndarray,
    mass: jnp.ndarray,
    temperature: jnp.ndarray,
) -> jnp.ndarray:
    """Assemble the drift-kinetic Coulomb operator for one species.

    The tabulated matrix is the full linearized Coulomb (Landau) collision
    operator projected onto the eight lowest Hermite-Laguerre moments, from
    Frei, Ernst & Ricci (2022), equations (C9a)--(C9f), generated at 80 decimal
    digits. Unlike the Sugama models it is only validated for like-species
    collisions, so a multi-species plasma is rejected rather than silently
    extrapolated.

    The tabulated coefficients are normalized to unit collision frequency; they
    are scaled here by the same dimensionless ``nu_aa = n / (sqrt(m) T**1.5)``
    used by the Sugama assemblers, so all three drift-kinetic models share one
    normalization.
    """

    density_s = jnp.atleast_1d(jnp.asarray(density))
    mass_s = jnp.atleast_1d(jnp.asarray(mass, dtype=jnp.result_type(density_s, float)))
    temperature_s = jnp.atleast_1d(
        jnp.asarray(temperature, dtype=jnp.result_type(density_s, mass_s, float))
    )
    if mass_s.shape != density_s.shape or temperature_s.shape != density_s.shape:
        raise ValueError("density, mass, and temperature must have equal length")
    for value, name in (
        (density_s, "density"),
        (mass_s, "mass"),
        (temperature_s, "temperature"),
    ):
        if not isinstance(value, jax.core.Tracer) and np.any(np.asarray(value) <= 0.0):
            raise ValueError(f"{name} must be positive")

    species_count = int(density_s.size)
    if species_count != 1:
        raise ValueError(
            "the tabulated drift-kinetic Coulomb operator is validated for "
            f"like-species collisions only, but {species_count} species were "
            "given; use collision_operator='sugama' or 'improved_sugama' for "
            "multispecies runs"
        )

    base = jnp.asarray(
        load_collision_moment_matrix("coulomb"),
        dtype=jnp.result_type(density_s, mass_s, temperature_s, float),
    )
    frequency = density_s[0] / (jnp.sqrt(mass_s[0]) * temperature_s[0] ** 1.5)
    return (frequency * base)[None, None, ...]


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


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class EqualSpeciesFiniteWavelengthSugamaOperator:
    r"""Finite-wavelength original/improved-Sugama tables for one species.

    The tabulated test and field matrices implement Frei et al. (2021),
    equations (3.72) and (3.79), optionally augmented by the improved field
    correction of Frei, Ernst & Ricci (2022), on the post-field nonadiabatic
    response :math:`H`. Unlike the Coulomb table, these models have no
    separately tabulated electrostatic polarization vectors.
    """

    bessel_argument_grid: jnp.ndarray
    pair_frequency: jnp.ndarray
    test_table: jnp.ndarray
    field_table: jnp.ndarray

    def apply(self, context: CollisionContext) -> jnp.ndarray:
        """Interpolate and apply the equal-species original-Sugama matrix."""

        state = jnp.asarray(context.hamiltonian)
        species_count = 1 if state.ndim == 5 else int(state.shape[0])
        if species_count != 1:
            raise ValueError(
                "equal-species finite-wavelength Sugama tables require one species"
            )
        bessel_argument = jnp.sqrt(2.0 * jnp.maximum(jnp.asarray(context.cache.b), 0.0))
        if bessel_argument.ndim < 1 or int(bessel_argument.shape[0]) != 1:
            raise ValueError("collision Bessel argument must have one species axis")
        test, field = (
            interpolate_collision_diagonal_table(
                self.bessel_argument_grid, table, bessel_argument[0]
            )
            for table in (self.test_table, self.field_table)
        )
        frequency = jnp.asarray(self.pair_frequency)
        if frequency.shape != (1, 1):
            raise ValueError("pair_frequency must have shape (1, 1)")
        matrix = (frequency[0, 0] * (test + field))[None, None, ...]
        return apply_multispecies_collision_moment_matrix(state, matrix)

    def tree_flatten(self):
        return (
            self.bessel_argument_grid,
            self.pair_frequency,
            self.test_table,
            self.field_table,
        ), None

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
