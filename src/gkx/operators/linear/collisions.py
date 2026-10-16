"""Linearized collision matrices, interpolation, and runtime application."""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import lru_cache, partial
import hashlib
import io
import json
from importlib import resources
from typing import TYPE_CHECKING, Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from gkx.operators.collision import CollisionContext

_COLLISION_MATRIX_DATA = "advanced_collision_six_moment.npy"
_COLLISION_MATRIX_METADATA = "advanced_collision_six_moment.json"
# ``(Nl, Nm) = (J+1, P+1)`` of the P=3, J=1 Appendix C truncation (Frei, Ernst
# & Ricci 2022); the transposed (4, 2) would pair coefficients with wrong moments.
DRIFT_KINETIC_MOMENT_LAYOUT: tuple[int, int] = (2, 4)

# Every moment contraction here goes through _exact_einsum. An unpinned dot may be
# lowered to TF32 on recent NVIDIA GPUs, which breaks the conservation identities
# these matrices carry at ~1e-4. No CPU value reveals it; the operator tests do.
_exact_einsum = partial(jnp.einsum, precision=jax.lax.Precision.HIGHEST)


@lru_cache(maxsize=1)
def _collision_matrix_bundle() -> tuple[np.ndarray, dict[str, Any]]:
    data_root = resources.files("gkx").joinpath("data")
    payload = data_root.joinpath(_COLLISION_MATRIX_DATA).read_bytes()
    metadata = json.loads(
        data_root.joinpath(_COLLISION_MATRIX_METADATA).read_text(encoding="utf-8")
    )
    digest = hashlib.sha256(payload).hexdigest()
    if digest != metadata.get("sha256"):
        raise ValueError("collision coefficient checksum does not match metadata")
    matrices = np.load(io.BytesIO(payload), allow_pickle=False)
    if list(matrices.shape) != metadata.get("shape"):
        raise ValueError("collision coefficient shape does not match metadata")
    if (metadata.get("Nl"), metadata.get("Nm")) != DRIFT_KINETIC_MOMENT_LAYOUT:
        raise ValueError("collision coefficient moment layout does not match metadata")
    return np.asarray(matrices), metadata


def load_collision_moment_matrix(model: str) -> np.ndarray:
    """Load a provenance-checked drift-kinetic collision moment matrix."""

    matrices, metadata = _collision_matrix_bundle()
    names = list(metadata["models"])
    key = model.strip().lower()
    if key not in names:
        raise ValueError(f"collision model must be one of {names}")
    return np.array(matrices[names.index(key)], copy=True)


def parallel_electric_field_source(
    maximum_hermite_order: int,
    maximum_laguerre_order: int,
    normalized_field: jnp.ndarray,
) -> jnp.ndarray:
    r"""Return the linearized parallel-electric-field gyro-moment source.

    Frei, Ernst & Ricci (2022), equation (81), gives the force ladder
    :math:`E\sqrt{2p}N^{p-1,j}`.  Linearizing about a Maxwellian leaves only
    the ``(p, j) = (1, 0)`` source in ``dN/dt = C N + s``.  The normalized
    field is :math:`eE/(v_{Te}m_e)` in the equation's convention, including
    its sign; for electrons in a positive physical field it is positive and
    the source is negative.
    """

    if maximum_hermite_order < 1:
        raise ValueError("maximum_hermite_order must be >= 1")
    if maximum_laguerre_order < 0:
        raise ValueError("maximum_laguerre_order must be >= 0")
    field = jnp.asarray(normalized_field)
    if field.ndim != 0:
        raise ValueError("normalized_field must be a scalar")
    n_laguerre = maximum_laguerre_order + 1
    source = jnp.zeros(
        ((maximum_hermite_order + 1) * n_laguerre,),
        dtype=jnp.result_type(field, float),
    )
    return source.at[n_laguerre].set(-jnp.sqrt(2.0) * field)


def solve_driven_collision_response(
    collision_matrix: jnp.ndarray,
    source: jnp.ndarray,
    *,
    active_modes: tuple[int, ...],
) -> jnp.ndarray:
    r"""Solve the constrained steady response ``C N + source = 0``.

    ``active_modes`` removes exact collision invariants and any intentionally
    truncated moments before the dense solve.  The returned full vector is
    zero outside that subspace.  The solve remains in JAX so currents and
    transport coefficients can be differentiated with respect to collision
    coefficients, species parameters, and the applied drive.
    """

    matrix = jnp.asarray(collision_matrix)
    drive = jnp.asarray(source, dtype=jnp.result_type(matrix, source))
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("collision_matrix must be square")
    if drive.shape != (matrix.shape[0],):
        raise ValueError("source must match the collision-matrix dimension")
    if not active_modes:
        raise ValueError("active_modes must contain at least one mode")
    if len(set(active_modes)) != len(active_modes):
        raise ValueError("active_modes must be unique")
    if min(active_modes) < 0 or max(active_modes) >= matrix.shape[0]:
        raise ValueError("active_modes contains an out-of-range index")

    indices = jnp.asarray(active_modes, dtype=jnp.int32)
    reduced = matrix[indices[:, None], indices[None, :]]
    response = -jnp.linalg.solve(reduced, drive[indices])
    return jnp.zeros_like(drive).at[indices].set(response)


def drift_kinetic_sugama_pair_matrices(
    mass_ratio: jnp.ndarray,
    temperature_ratio: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return normalized test/field matrices for one ordered species pair.

    This is Frei, Ernst & Ricci (2022), Appendix C, equations (C4)--(C5),
    with ``mass_ratio = m_a / m_b`` and ``temperature_ratio = T_a / T_b``.
    Multiplication by the directed collision frequency ``nu_ab`` is left to the
    caller. The matrices use the code's Hermite-major, signed-Laguerre ordering.
    """

    sigma = jnp.asarray(mass_ratio)
    tau = jnp.asarray(temperature_ratio, dtype=jnp.result_type(sigma, float))
    if sigma.ndim != 0 or tau.ndim != 0:
        raise ValueError("mass_ratio and temperature_ratio must be scalars")
    if not isinstance(sigma, jax.core.Tracer) and float(sigma) <= 0.0:
        raise ValueError("mass_ratio must be positive")
    if not isinstance(tau, jax.core.Tracer) and float(tau) <= 0.0:
        raise ValueError("temperature_ratio must be positive")
    dtype = jnp.result_type(sigma, tau)
    pi = jnp.asarray(jnp.pi, dtype=dtype)
    root_pi = jnp.sqrt(pi)
    total = sigma + tau
    test = jnp.zeros((8, 8), dtype=dtype)
    field = jnp.zeros((8, 8), dtype=dtype)
    mode_01, mode_10, mode_11, mode_20, mode_30 = 1, 2, 3, 4, 6

    test = test.at[mode_10, mode_10].set(
        -8.0 * (sigma + 1.0) / (3.0 * root_pi) * (tau / total) ** 1.5
    )
    test_10_30 = (
        4.0 / 5.0 * jnp.sqrt(6.0 / pi) * jnp.sqrt(sigma + 1.0) * tau**2 / total**2
    )
    test_10_11 = -8.0 * jnp.sqrt(sigma + 1.0) * tau**2 / (5.0 * root_pi * total**2)
    test = test.at[mode_10, mode_30].set(test_10_30)
    test = test.at[mode_30, mode_10].set(test_10_30)
    test = test.at[mode_10, mode_11].set(test_10_11)
    test = test.at[mode_11, mode_10].set(test_10_11)
    test = test.at[mode_20, mode_20].set(
        -16.0
        * jnp.sqrt(tau)
        * (5.0 * sigma**2 * (tau + 2.0) + 21.0 * sigma * tau + 6.0 * tau**2)
        / (45.0 * root_pi * total**2.5)
    )
    test_20_01 = (
        -16.0
        * jnp.sqrt(2.0 / pi)
        * jnp.sqrt(tau)
        * (-5.0 * sigma**2 * (tau - 1.0) + 3.0 * sigma * tau + 3.0 * tau**2)
        / (45.0 * total**2.5)
    )
    test = test.at[mode_20, mode_01].set(test_20_01)
    test = test.at[mode_01, mode_20].set(test_20_01)
    test = test.at[mode_01, mode_01].set(
        -16.0
        * jnp.sqrt(tau)
        * (5.0 * sigma**2 + 2.0 * (5.0 * sigma + 9.0) * sigma * tau + 3.0 * tau**2)
        / (45.0 * root_pi * total**2.5)
    )
    test = test.at[mode_30, mode_30].set(
        -4.0
        * jnp.sqrt(tau)
        * (70.0 * sigma**2 + 56.0 * sigma * tau + 31.0 * tau**2)
        / (35.0 * root_pi * total**2.5)
    )
    test_30_11 = (
        -4.0
        * jnp.sqrt(2.0 / (3.0 * pi))
        * tau**1.5
        * (28.0 * sigma + tau)
        / (35.0 * total**2.5)
    )
    test = test.at[mode_30, mode_11].set(test_30_11)
    test = test.at[mode_11, mode_30].set(test_30_11)
    test = test.at[mode_11, mode_11].set(
        -8.0
        * jnp.sqrt(tau)
        * (105.0 * sigma**2 + 98.0 * sigma * tau + 47.0 * tau**2)
        / (105.0 * root_pi * total**2.5)
    )

    field = field.at[mode_10, mode_10].set(
        8.0 * sigma * (sigma + 1.0) * tau / (3.0 * root_pi * jnp.sqrt(sigma * total**3))
    )
    field = field.at[mode_10, mode_30].set(
        -4.0
        * jnp.sqrt(6.0 / pi)
        * jnp.sqrt(sigma**3 * (sigma + 1.0))
        * tau
        / (5.0 * total**2)
    )
    field = field.at[mode_10, mode_11].set(
        8.0 * jnp.sqrt(sigma**3 * (sigma + 1.0)) * tau / (5.0 * root_pi * total**2)
    )
    field_20_01 = (
        -16.0
        * jnp.sqrt(2.0 / pi)
        * sigma
        * (sigma + 1.0)
        * tau**1.5
        / (9.0 * total**2.5)
    )
    field = field.at[mode_20, mode_20].set(
        16.0 * sigma * (sigma + 1.0) * tau**1.5 / (9.0 * root_pi * total**2.5)
    )
    field = field.at[mode_20, mode_01].set(field_20_01)
    field = field.at[mode_01, mode_20].set(field_20_01)
    field = field.at[mode_01, mode_01].set(
        32.0 * sigma * (sigma + 1.0) * tau**1.5 / (9.0 * root_pi * total**2.5)
    )
    field = field.at[mode_30, mode_10].set(
        -4.0
        * jnp.sqrt(6.0 / pi)
        * jnp.sqrt(sigma * (sigma + 1.0) * tau**3)
        / (5.0 * total**2)
    )
    field = field.at[mode_30, mode_30].set(
        36.0 * (sigma * tau) ** 1.5 / (25.0 * root_pi * total**2.5)
    )
    field_30_11 = (
        -12.0 * jnp.sqrt(6.0 / pi) * (sigma * tau) ** 1.5 / (25.0 * total**2.5)
    )
    field = field.at[mode_30, mode_11].set(field_30_11)
    field = field.at[mode_11, mode_10].set(
        8.0 * jnp.sqrt(sigma * (sigma + 1.0) * tau**3) / (5.0 * root_pi * total**2)
    )
    field = field.at[mode_11, mode_30].set(field_30_11)
    field = field.at[mode_11, mode_11].set(
        24.0 * (sigma * tau) ** 1.5 / (25.0 * root_pi * total**2.5)
    )

    laguerre_sign = jnp.asarray(
        [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=dtype
    )
    convention = laguerre_sign[:, None] * laguerre_sign[None, :]
    return convention * test, convention * field


def drift_kinetic_improved_sugama_pair_matrices(
    mass_ratio: jnp.ndarray,
    temperature_ratio: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the lowest-order improved-Sugama test and field matrices.

    The correction is Frei, Ernst & Ricci (2022), Appendix C, equations
    (101)--(102), added to the original-Sugama ordered-pair matrices. As for
    :func:`drift_kinetic_sugama_pair_matrices`, the directed frequency is not
    included and the result uses the code's signed-Laguerre convention.
    """

    original_test, original_field = drift_kinetic_sugama_pair_matrices(
        mass_ratio, temperature_ratio
    )
    sigma = jnp.asarray(mass_ratio)
    tau = jnp.asarray(temperature_ratio, dtype=original_test.dtype)
    pi = jnp.asarray(jnp.pi, dtype=original_test.dtype)
    root_pi = jnp.sqrt(pi)
    total = sigma + tau
    root_tau = jnp.sqrt(tau)
    ratio_root = jnp.sqrt((sigma + 1.0) * tau / total)
    test = jnp.zeros((8, 8), dtype=original_test.dtype)
    field = jnp.zeros((8, 8), dtype=original_test.dtype)
    mode_10, mode_11, mode_30 = 2, 3, 6

    test = test.at[mode_30, mode_10].set(
        4.0
        * tau**1.5
        * jnp.sqrt(6.0 / pi)
        * (tau * (-ratio_root + sigma + 1.0) - sigma * ratio_root)
        / (5.0 * total**2.5)
    )
    test = test.at[mode_11, mode_10].set(
        8.0
        * tau**1.5
        * (sigma * ratio_root + tau * (ratio_root - sigma - 1.0))
        / (5.0 * root_pi * total**2.5)
    )
    mixed_polynomial = (
        10.0 * sigma**2 * (tau - 1.0)
        + 3.0 * jnp.sqrt((sigma + 1.0) * tau**5 / total)
        + sigma * tau * (3.0 * ratio_root + tau - 4.0)
        - 3.0 * tau**2
    )
    test = test.at[mode_10, mode_30].set(
        -4.0
        * root_tau
        * jnp.sqrt(2.0 / (3.0 * pi))
        * mixed_polynomial
        / (5.0 * total**2.5)
    )
    diagonal_polynomial = (
        sigma
        * (tau - 1.0)
        * root_tau
        * (10.0 * sigma**2 - 2.0 * sigma * tau + 3.0 * tau**2)
    )
    test = test.at[mode_30, mode_30].set(
        -12.0 * diagonal_polynomial / (25.0 * root_pi * total**3.5)
    )
    test_11_30 = 4.0 * jnp.sqrt(6.0 / pi) * diagonal_polynomial / (25.0 * total**3.5)
    test = test.at[mode_11, mode_30].set(test_11_30)
    test = test.at[mode_10, mode_11].set(
        8.0 * root_tau * mixed_polynomial / (15.0 * root_pi * total**2.5)
    )
    test = test.at[mode_30, mode_11].set(test_11_30)
    test = test.at[mode_11, mode_11].set(
        -8.0 * diagonal_polynomial / (25.0 * root_pi * total**3.5)
    )

    root_total = jnp.sqrt(total)
    root_sigma_tau = jnp.sqrt(sigma * tau)
    root_sigma_plus_one = jnp.sqrt(sigma + 1.0)
    field = field.at[mode_30, mode_10].set(
        -4.0
        * jnp.sqrt(6.0 / pi)
        * sigma**1.5
        * tau
        * (-jnp.sqrt((sigma + 1.0) * total) + sigma + 1.0)
        / (5.0 * total**2.5)
    )
    field = field.at[mode_11, mode_10].set(
        8.0
        * sigma
        * root_sigma_tau
        * (
            -jnp.sqrt((sigma + 1.0) * tau**3)
            + sigma * (jnp.sqrt(tau * total) - jnp.sqrt((sigma + 1.0) * tau))
            + jnp.sqrt(tau * total)
        )
        / (5.0 * root_pi * total**3)
    )
    field = field.at[mode_10, mode_30].set(
        4.0
        * jnp.sqrt(6.0 / pi)
        * jnp.sqrt(sigma)
        * tau
        * (
            jnp.sqrt((sigma + 1.0) * tau**3)
            - tau * root_total
            + sigma
            * (
                -3.0 * tau * root_total
                + jnp.sqrt((sigma + 1.0) * tau)
                + 2.0 * root_total
            )
        )
        / (5.0 * total**3)
    )
    field_diagonal = sigma * (5.0 * tau - root_tau - 2.0) - (root_tau - 3.0) * tau
    field = field.at[mode_30, mode_30].set(
        36.0 * sigma**1.5 * tau * field_diagonal / (25.0 * root_pi * total**3.5)
    )
    field_11_30 = (
        12.0
        * jnp.sqrt(6.0 / pi)
        * sigma**1.5
        * tau
        * (sigma * (-5.0 * tau + root_tau + 2.0) + (root_tau - 3.0) * tau)
        / (25.0 * total**3.5)
    )
    field = field.at[mode_11, mode_30].set(field_11_30)
    field = field.at[mode_10, mode_11].set(
        -8.0
        * root_sigma_tau
        * (
            sigma
            * (
                -3.0 * jnp.sqrt(tau**3 * total)
                + root_sigma_plus_one * tau
                + 2.0 * jnp.sqrt(tau * total)
            )
            - jnp.sqrt(tau**3 * total)
            + root_sigma_plus_one * tau**2
        )
        / (5.0 * root_pi * total**3)
    )
    field = field.at[mode_30, mode_11].set(field_11_30)
    field = field.at[mode_11, mode_11].set(
        24.0 * sigma**1.5 * tau * field_diagonal / (25.0 * root_pi * total**3.5)
    )

    laguerre_sign = jnp.asarray(
        [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
        dtype=original_test.dtype,
    )
    convention = laguerre_sign[:, None] * laguerre_sign[None, :]
    # Appendix C writes the driven moment as the superscript and the response
    # moment as the subscript, opposite to the row/column application contract.
    return original_test + convention * test.T, original_field + convention * field.T


def _assemble_drift_kinetic_sugama_matrix(
    density: jnp.ndarray,
    mass: jnp.ndarray,
    temperature: jnp.ndarray,
    pair_matrices: Callable[
        [jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ],
    pair_scale: jnp.ndarray | float = 1.0,
) -> jnp.ndarray:
    density_s = jnp.asarray(density)
    mass_s = jnp.asarray(mass, dtype=jnp.result_type(density_s, float))
    temperature_s = jnp.asarray(
        temperature, dtype=jnp.result_type(density_s, mass_s, float)
    )
    if density_s.ndim != 1 or mass_s.shape != density_s.shape:
        raise ValueError("density and mass must be one-dimensional with equal length")
    if temperature_s.shape != density_s.shape:
        raise ValueError("temperature must have the same shape as density")
    for value, name in (
        (density_s, "density"),
        (mass_s, "mass"),
        (temperature_s, "temperature"),
    ):
        if not isinstance(value, jax.core.Tracer) and np.any(np.asarray(value) <= 0.0):
            raise ValueError(f"{name} must be positive")

    ns = int(density_s.size)
    mass_ratio = mass_s[:, None] / mass_s[None, :]
    temperature_ratio = temperature_s[:, None] / temperature_s[None, :]
    pair_function = jax.vmap(pair_matrices)
    test, field = pair_function(mass_ratio.reshape(-1), temperature_ratio.reshape(-1))
    test = test.reshape(ns, ns, 8, 8)
    field = field.reshape(ns, ns, 8, 8)
    frequency = (
        pair_scale
        * density_s[None, :]
        / (jnp.sqrt(mass_s[:, None]) * temperature_s[:, None] ** 1.5)
    )
    matrix = frequency[..., None, None] * field
    diagonal = jnp.arange(ns)
    return matrix.at[diagonal, diagonal].add(
        jnp.sum(frequency[..., None, None] * test, axis=1)
    )


def assemble_drift_kinetic_sugama_matrix(
    density: jnp.ndarray,
    mass: jnp.ndarray,
    temperature: jnp.ndarray,
) -> jnp.ndarray:
    """Assemble the normalized original-Sugama operator for all species.

    The returned axes are ``(target species, source species, target moment,
    source moment)``. Collision frequencies use the dimensionless scaling
    ``nu_ab = n_b / (sqrt(m_a) T_a**(3/2))``; callers remain responsible for
    any common dimensional prefactor.
    """

    return _assemble_drift_kinetic_sugama_matrix(
        density, mass, temperature, drift_kinetic_sugama_pair_matrices
    )


def assemble_drift_kinetic_improved_sugama_matrix(
    density: jnp.ndarray,
    mass: jnp.ndarray,
    temperature: jnp.ndarray,
) -> jnp.ndarray:
    """Assemble the normalized lowest-order improved-Sugama species matrix."""

    return _assemble_drift_kinetic_sugama_matrix(
        density, mass, temperature, drift_kinetic_improved_sugama_pair_matrices
    )


class _FieldPytree:
    """Flatten a frozen dataclass into its fields, in declaration order."""

    def tree_flatten(self):
        return tuple(getattr(self, f.name) for f in fields(self)), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DriftKineticMomentCollisionOperator(_FieldPytree):
    r"""Dense drift-kinetic collision matrix acting on gyrocenter moments.

    In the zero-Larmor-radius limit the particle perturbation satisfies
    :math:`f\simeq g`; the Hermite--Laguerre matrix therefore acts on the
    evolved gyrocenter distribution, not on the post-field Hamiltonian
    :math:`H=g+qF_M\phi/T`. See Frei, Ernst & Ricci (2022), equation (73).
    """

    matrix: jnp.ndarray

    @classmethod
    def from_species(
        cls,
        density: jnp.ndarray,
        mass: jnp.ndarray,
        temperature: jnp.ndarray,
    ) -> DriftKineticMomentCollisionOperator:
        """Build the ordered-pair matrix from physical species parameters."""

        return cls(assemble_drift_kinetic_sugama_matrix(density, mass, temperature))

    @classmethod
    def from_improved_species(
        cls,
        density: jnp.ndarray,
        mass: jnp.ndarray,
        temperature: jnp.ndarray,
    ) -> DriftKineticMomentCollisionOperator:
        """Build the lowest-order improved-Sugama matrix for all species."""

        return cls(
            assemble_drift_kinetic_improved_sugama_matrix(density, mass, temperature)
        )

    def apply(self, context: CollisionContext) -> jnp.ndarray:
        """Apply the drift-kinetic matrix to evolved gyrocenter moments."""

        return apply_multispecies_collision_moment_matrix(
            context.distribution, self.matrix
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FiniteWavelengthCoulombOperator(_FieldPytree):
    """Tabulated finite-wavelength Coulomb test, field, and polarization blocks.

    Pair tables have independent target/source Bessel-argument axes
    ``B = kperp*v_thermal/Omega``. Since the runtime cache stores
    ``b = kperp**2*T*m/(q*B_ref)**2``, the interpolation coordinate is
    ``B = sqrt(2*b)``. Their matrices act on gyrocenter moments ``G``;
    polarization vectors supply the particle-to-gyrocenter pullback terms from
    Frei et al. (2021), equation (3.50), without double-counting ``build_H``.
    """

    bessel_argument_grid: jnp.ndarray
    pair_frequency: jnp.ndarray
    test_table: jnp.ndarray
    field_table: jnp.ndarray
    test_phi1: jnp.ndarray
    field_phi1: jnp.ndarray
    test_phi2: jnp.ndarray
    field_phi2: jnp.ndarray

    def apply(self, context: CollisionContext) -> jnp.ndarray:
        """Apply the resolved operator using the solved electrostatic field."""

        inverse_tz = jnp.where(
            jnp.asarray(context.parameters.tz) == 0.0,
            0.0,
            1.0 / jnp.asarray(context.parameters.tz),
        )
        bessel_argument = jnp.sqrt(2.0 * jnp.maximum(jnp.asarray(context.cache.b), 0.0))
        resolved = tuple(
            interpolate_collision_pair_table(
                self.bessel_argument_grid, table, bessel_argument
            )
            for table in (
                self.test_table,
                self.field_table,
                self.test_phi1,
                self.field_phi1,
                self.test_phi2,
                self.field_phi2,
            )
        )
        return apply_finite_wavelength_coulomb_moment_operator(
            context.distribution,
            *resolved,
            phi=context.fields.phi,
            pair_frequency=self.pair_frequency,
            charge_over_temperature=inverse_tz,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class EqualSpeciesFiniteWavelengthCoulombOperator(_FieldPytree):
    """Finite-wavelength Coulomb tables for one equal-species plasma.

    Like-species collisions have the same target and source Bessel argument at
    every spatial point. Storing only that diagonal avoids the quadratic
    target/source wavelength table while retaining the complete test, field,
    and polarization terms. The interpolation remains differentiable in JAX.
    """

    bessel_argument_grid: jnp.ndarray
    pair_frequency: jnp.ndarray
    test_table: jnp.ndarray
    field_table: jnp.ndarray
    test_phi1: jnp.ndarray
    field_phi1: jnp.ndarray
    test_phi2: jnp.ndarray
    field_phi2: jnp.ndarray

    def apply(self, context: CollisionContext) -> jnp.ndarray:
        """Interpolate the diagonal table and apply it to one species."""

        distribution = jnp.asarray(context.distribution)
        species_count = 1 if distribution.ndim == 5 else int(distribution.shape[0])
        if species_count != 1:
            raise ValueError(
                "equal-species finite-wavelength Coulomb tables require one species"
            )
        bessel_argument = jnp.sqrt(2.0 * jnp.maximum(jnp.asarray(context.cache.b), 0.0))
        if bessel_argument.ndim < 1 or int(bessel_argument.shape[0]) != 1:
            raise ValueError("collision Bessel argument must have one species axis")
        resolved = tuple(
            interpolate_collision_diagonal_table(
                self.bessel_argument_grid, table, bessel_argument[0]
            )[None, None, ...]
            for table in (
                self.test_table,
                self.field_table,
                self.test_phi1,
                self.field_phi1,
                self.test_phi2,
                self.field_phi2,
            )
        )
        inverse_tz = jnp.where(
            jnp.asarray(context.parameters.tz) == 0.0,
            0.0,
            1.0 / jnp.asarray(context.parameters.tz),
        )
        return apply_finite_wavelength_coulomb_moment_operator(
            distribution,
            *resolved,
            phi=context.fields.phi,
            pair_frequency=self.pair_frequency,
            charge_over_temperature=inverse_tz,
        )


def interpolate_collision_diagonal_table(
    bessel_argument_grid: jnp.ndarray,
    table: jnp.ndarray,
    bessel_argument: jnp.ndarray,
) -> jnp.ndarray:
    """Interpolate one equal-species vector or matrix table along its diagonal.

    ``table`` has shape ``(B, moment)`` or ``(B, output_moment, input_moment)``.
    Coefficient axes are moved before the spatial axes in the returned array.
    Values outside the tabulated interval use the nearest endpoint.
    """

    grid = jnp.asarray(bessel_argument_grid)
    coefficients = jnp.asarray(table)
    target = jnp.asarray(bessel_argument, dtype=jnp.result_type(grid, coefficients))
    if grid.ndim != 1 or int(grid.size) < 2:
        raise ValueError(
            "collision Bessel-argument grid must be one-dimensional with at least two points"
        )
    if coefficients.ndim not in {2, 3}:
        raise ValueError(
            "diagonal collision table must contain one vector or two matrix axes"
        )
    if int(coefficients.shape[0]) != int(grid.size):
        raise ValueError("diagonal collision table axis must match the grid")
    if coefficients.ndim == 3 and int(coefficients.shape[-1]) != int(
        coefficients.shape[-2]
    ):
        raise ValueError("diagonal collision matrices must be square")
    if not isinstance(grid, jax.core.Tracer):
        host_grid = np.asarray(grid)
        if not np.all(np.isfinite(host_grid)) or not np.all(np.diff(host_grid) > 0.0):
            raise ValueError(
                "collision Bessel-argument grid must be finite and strictly increasing"
            )

    clipped = jnp.clip(target, grid[0], grid[-1])
    left = jnp.clip(jnp.searchsorted(grid, clipped, side="right") - 1, 0, grid.size - 2)
    fraction = (clipped - grid[left]) / (grid[left + 1] - grid[left])
    coefficient_ndim = coefficients.ndim - 1
    weight = fraction[(...,) + (None,) * coefficient_ndim]
    interpolated = coefficients[left] + weight * (
        coefficients[left + 1] - coefficients[left]
    )
    spatial_ndim = target.ndim
    return jnp.moveaxis(
        interpolated,
        tuple(range(spatial_ndim, spatial_ndim + coefficient_ndim)),
        tuple(range(coefficient_ndim)),
    )


def interpolate_collision_pair_table(
    kperp_grid: jnp.ndarray,
    tables: jnp.ndarray,
    kperp: jnp.ndarray,
) -> jnp.ndarray:
    """Bilinearly interpolate target/source finite-wavelength pair tables.

    ``tables`` has shape ``(target, source, target_k, source_k, coefficients...)``
    with one vector or two matrix coefficient axes. ``kperp`` has a leading
    species axis followed by the simulation spatial axes. The result moves the
    coefficient axes ahead of the spatial axes for direct moment contraction.
    """

    grid = jnp.asarray(kperp_grid)
    table = jnp.asarray(tables)
    target = jnp.asarray(kperp, dtype=jnp.result_type(grid, table))
    if grid.ndim != 1 or int(grid.size) < 2:
        raise ValueError(
            "collision kperp grid must be one-dimensional with at least two points"
        )
    if table.ndim not in {5, 6}:
        raise ValueError(
            "collision pair table must have target/source species, target/source "
            "kperp, and one vector or two matrix coefficient axes"
        )
    ns = int(table.shape[0])
    if int(table.shape[1]) != ns:
        raise ValueError("collision pair table must have equal target/source axes")
    if int(table.shape[2]) != int(grid.size) or int(table.shape[3]) != int(grid.size):
        raise ValueError("collision pair table kperp axes must match the grid")
    if table.ndim == 6 and int(table.shape[-1]) != int(table.shape[-2]):
        raise ValueError("collision pair matrices must be square")
    if target.ndim < 1 or int(target.shape[0]) != ns:
        raise ValueError("kperp must have a leading axis matching the species count")
    if not isinstance(grid, jax.core.Tracer):
        host_grid = np.asarray(grid)
        if not np.all(np.isfinite(host_grid)) or not np.all(np.diff(host_grid) > 0.0):
            raise ValueError(
                "collision kperp grid must be finite and strictly increasing"
            )

    coefficient_ndim = table.ndim - 4

    def interpolation_coordinates(values: jnp.ndarray):
        clipped = jnp.clip(values, grid[0], grid[-1])
        left = jnp.clip(
            jnp.searchsorted(grid, clipped, side="right") - 1, 0, grid.size - 2
        )
        fraction = (clipped - grid[left]) / (grid[left + 1] - grid[left])
        return left, fraction

    def interpolate_one(
        pair_table: jnp.ndarray,
        target_values: jnp.ndarray,
        source_values: jnp.ndarray,
    ) -> jnp.ndarray:
        target_left, target_fraction = interpolation_coordinates(target_values)
        source_left, source_fraction = interpolation_coordinates(source_values)
        target_weight = target_fraction[(...,) + (None,) * coefficient_ndim]
        source_weight = source_fraction[(...,) + (None,) * coefficient_ndim]
        lower_source = pair_table[target_left, source_left]
        upper_source = pair_table[target_left, source_left + 1]
        lower_target = lower_source + source_weight * (upper_source - lower_source)
        lower_source = pair_table[target_left + 1, source_left]
        upper_source = pair_table[target_left + 1, source_left + 1]
        upper_target = lower_source + source_weight * (upper_source - lower_source)
        interpolated = lower_target + target_weight * (upper_target - lower_target)
        spatial_ndim = target_values.ndim
        return jnp.moveaxis(
            interpolated,
            tuple(range(spatial_ndim, spatial_ndim + coefficient_ndim)),
            tuple(range(coefficient_ndim)),
        )

    def interpolate_sources(
        target_tables: jnp.ndarray,
        target_values: jnp.ndarray,
    ) -> jnp.ndarray:
        return jax.vmap(
            lambda pair_table, source_values: interpolate_one(
                pair_table, target_values, source_values
            )
        )(target_tables, target)

    return jax.vmap(interpolate_sources)(table, target)


def apply_collision_moment_matrix(
    state: jnp.ndarray,
    matrix: jnp.ndarray,
    *,
    nu: jnp.ndarray,
    weight: jnp.ndarray = jnp.asarray(1.0),
) -> jnp.ndarray:
    """Apply a dense drift-kinetic matrix in Hermite-major moment ordering."""

    value = jnp.asarray(state)
    if value.ndim not in {5, 6}:
        raise ValueError("collision state must have five or six dimensions")
    expanded = value[None, ...] if value.ndim == 5 else value
    ns, nl, nm = map(int, expanded.shape[:3])
    mode_count = nl * nm
    coefficients = jnp.asarray(matrix, dtype=jnp.result_type(value, matrix))
    spatial_shape = tuple(map(int, expanded.shape[3:]))
    static_shared = (mode_count, mode_count)
    static_species = {(1,) + static_shared, (ns,) + static_shared}
    spatial_shared = static_shared + spatial_shape
    spatial_species = {(1,) + spatial_shared, (ns,) + spatial_shared}
    if coefficients.shape == static_shared or coefficients.shape == spatial_shared:
        coefficients = coefficients[None, ...]
    if coefficients.shape not in static_species | spatial_species:
        raise ValueError(
            "collision matrix must have shape for a static or state-spatial operator, "
            "with an optional leading species axis"
        )
    coefficients = jnp.broadcast_to(coefficients, (ns,) + coefficients.shape[1:])
    packed = jnp.swapaxes(expanded, 1, 2).reshape((ns, mode_count) + expanded.shape[3:])
    applied = _exact_einsum("sij...,sj...->si...", coefficients, packed)
    result = jnp.swapaxes(applied.reshape((ns, nm, nl) + expanded.shape[3:]), 1, 2)
    real_dtype = jnp.real(expanded).dtype
    frequency = jnp.asarray(nu, dtype=real_dtype).reshape(-1)
    if frequency.size == 1:
        frequency = jnp.broadcast_to(frequency, (ns,))
    if int(frequency.size) != ns:
        raise ValueError(f"nu must have length {ns} (got {frequency.size})")
    scale = frequency[(slice(None),) + (None,) * (result.ndim - 1)]
    result = jnp.asarray(weight, dtype=real_dtype) * scale * result
    return result[0] if value.ndim == 5 else result


def apply_multispecies_collision_moment_matrix(
    state: jnp.ndarray,
    matrix: jnp.ndarray,
    *,
    weight: jnp.ndarray = jnp.asarray(1.0),
) -> jnp.ndarray:
    """Apply a target/source-species collision matrix and sum source species."""

    value = jnp.asarray(state)
    if value.ndim not in {5, 6}:
        raise ValueError("collision state must have five or six dimensions")
    expanded = value[None, ...] if value.ndim == 5 else value
    ns, nl, nm = map(int, expanded.shape[:3])
    mode_count = nl * nm
    spatial_shape = tuple(map(int, expanded.shape[3:]))
    coefficients = jnp.asarray(matrix, dtype=jnp.result_type(value, matrix))
    allowed = {
        (ns, ns, mode_count, mode_count),
        (ns, ns, mode_count, mode_count) + spatial_shape,
    }
    if coefficients.shape not in allowed:
        raise ValueError(
            "multispecies collision matrix must have target/source species and "
            "square moment axes, optionally followed by the state spatial shape"
        )
    packed = jnp.swapaxes(expanded, 1, 2).reshape((ns, mode_count) + expanded.shape[3:])
    applied = _exact_einsum("stij...,tj...->si...", coefficients, packed)
    result = jnp.swapaxes(applied.reshape((ns, nm, nl) + expanded.shape[3:]), 1, 2)
    result = jnp.asarray(weight, dtype=jnp.real(expanded).dtype) * result
    return result[0] if value.ndim == 5 else result


def apply_finite_wavelength_coulomb_moment_operator(
    distribution: jnp.ndarray,
    test_matrix: jnp.ndarray,
    field_matrix: jnp.ndarray,
    test_phi1: jnp.ndarray,
    field_phi1: jnp.ndarray,
    test_phi2: jnp.ndarray,
    field_phi2: jnp.ndarray,
    *,
    phi: jnp.ndarray,
    pair_frequency: jnp.ndarray,
    charge_over_temperature: jnp.ndarray,
    weight: jnp.ndarray = jnp.asarray(1.0),
) -> jnp.ndarray:
    r"""Apply equations (3.47)--(3.50) on resolved moment coefficients.

    Matrix axes are ``(target, source, output moment, input moment, ...)``;
    vector axes are ``(target, source, output moment, ...)``. Optional trailing
    axes must equal the state's ``(ky, kx, z)`` shape. ``pair_frequency[a,b]``
    multiplies all test, field, and polarization contributions for the ordered
    collision pair. Moment ordering is Hermite-major, matching the generated
    tables.
    """

    value = jnp.asarray(distribution)
    if value.ndim not in {5, 6}:
        raise ValueError("Coulomb collision state must have five or six dimensions")
    expanded = value[None, ...] if value.ndim == 5 else value
    ns, nl, nm = map(int, expanded.shape[:3])
    mode_count = nl * nm
    spatial_shape = tuple(map(int, expanded.shape[3:]))
    real_dtype = jnp.real(expanded).dtype
    coefficient_dtype = jnp.result_type(value, test_matrix, field_matrix)

    matrices = tuple(
        jnp.asarray(matrix, dtype=coefficient_dtype)
        for matrix in (test_matrix, field_matrix)
    )
    matrix_shapes = {
        (ns, ns, mode_count, mode_count),
        (ns, ns, mode_count, mode_count) + spatial_shape,
    }
    if any(matrix.shape not in matrix_shapes for matrix in matrices):
        raise ValueError(
            "Coulomb test/field matrices must have target/source and square "
            "moment axes, optionally followed by the state spatial shape"
        )

    vectors = tuple(
        jnp.asarray(vector, dtype=coefficient_dtype)
        for vector in (test_phi1, field_phi1, test_phi2, field_phi2)
    )
    vector_shapes = {
        (ns, ns, mode_count),
        (ns, ns, mode_count) + spatial_shape,
    }
    if any(vector.shape not in vector_shapes for vector in vectors):
        raise ValueError(
            "Coulomb polarization vectors must have target/source and moment "
            "axes, optionally followed by the state spatial shape"
        )

    frequency = jnp.asarray(pair_frequency, dtype=real_dtype)
    if frequency.shape != (ns, ns):
        raise ValueError(f"pair_frequency must have shape ({ns}, {ns})")
    charge_temperature = jnp.asarray(charge_over_temperature, dtype=real_dtype).reshape(
        -1
    )
    if int(charge_temperature.size) != ns:
        raise ValueError(f"charge_over_temperature must have length {ns}")
    potential = jnp.asarray(phi, dtype=coefficient_dtype)
    if potential.shape != spatial_shape:
        raise ValueError(f"phi must have spatial shape {spatial_shape}")

    packed = jnp.swapaxes(expanded, 1, 2).reshape((ns, mode_count) + expanded.shape[3:])
    test_applied = _exact_einsum("stij...,sj...->sti...", matrices[0], packed)
    field_applied = _exact_einsum("stij...,tj...->sti...", matrices[1], packed)
    pair_axes = (None,) * (test_applied.ndim - 2)
    frequency_view = frequency[(slice(None), slice(None)) + pair_axes]
    particle = jnp.sum(frequency_view * (test_applied + field_applied), axis=1)

    vector_pair_axes = (None,) * (vectors[0].ndim - 2)
    target_charge = charge_temperature[(slice(None), None) + vector_pair_axes]
    source_charge = charge_temperature[(None, slice(None)) + vector_pair_axes]
    polarization = target_charge * (vectors[0] + vectors[2]) + source_charge * (
        vectors[1] + vectors[3]
    )
    polarization = jnp.sum(
        frequency[(slice(None), slice(None)) + vector_pair_axes] * polarization,
        axis=1,
    )
    if vectors[0].ndim == 3:
        polarization = polarization[
            (slice(None), slice(None)) + (None,) * len(spatial_shape)
        ]
    polarization = polarization * potential[None, None, ...]

    result_packed = particle + polarization
    result = jnp.swapaxes(
        result_packed.reshape((ns, nm, nl) + expanded.shape[3:]), 1, 2
    )
    result = jnp.asarray(weight, dtype=real_dtype) * result
    return result[0] if value.ndim == 5 else result


def _landau_pair_frequency(nu: Any, density: Any, charge: Any, ns: int, dtype: Any):
    r"""Return :math:`\nu_{ab} = \nu_a n_b Z_b^2 / (n_a Z_a^2)` as ``(s, s)``.

    ``nu`` is each species' self-collision frequency; inter-species
    conservation is exact when the ``nu_a`` are mutually consistent
    (:math:`\nu_a \propto n_a Z_a^4 / (m_a^{1/2} T_a^{3/2})`).
    """

    rate = jnp.broadcast_to(jnp.asarray(nu, dtype=dtype).reshape(-1), (ns,))
    weight = jnp.broadcast_to(
        jnp.asarray(density, dtype=dtype).reshape(-1)
        * jnp.asarray(charge, dtype=dtype).reshape(-1) ** 2,
        (ns,),
    )
    return rate[:, None] * weight[None, :] / weight[:, None]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LandauBlocks:
    """Block-sparse factors of the linearized Landau operator, ``T B T^T``.

    ``T`` (Hermite-Laguerre to Legendre-Sonine) is block diagonal in the degree
    ``p + 2j`` and ``B`` in the Legendre index ``l``, so the dense product is
    replaced by small batched ones: 4.1x fewer flops at ``(Nl, Nm) = (16, 48)``.
    Blocks of similar size share one padded batch; padded indices point at one
    extra zero row. ``*_diagonal`` hold the per-pair diagonal of the dense
    operator for preconditioners.
    """

    transform_rows: tuple  # per batch (blocks, R) Hermite-Laguerre indices
    transform_cols: tuple  # per batch (blocks, C) Legendre-Sonine indices
    transform: tuple  # per batch (blocks, R, C)
    block_index: tuple  # per batch (blocks, K) Legendre-Sonine indices
    block_test: tuple  # per batch (s, s, blocks, K, K)
    block_field: tuple  # per batch (s, s, blocks, K, K)
    test_diagonal: jnp.ndarray  # (s, s, n)
    field_diagonal: jnp.ndarray  # (s, s, n)
    sonine_count: int  # Legendre-Sonine functions, static

    def tree_flatten(self):
        return tuple(getattr(self, f.name) for f in fields(self))[
            :-1
        ], self.sonine_count

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, aux_data)

    def diagonal(self, **frequencies: Any) -> jnp.ndarray:
        """Return the ``(s, n)`` diagonal of the assembled operator."""

        ns = int(self.test_diagonal.shape[0])
        pair = _landau_pair_frequency(
            ns=ns, dtype=self.test_diagonal.dtype, **frequencies
        )[..., None]
        own = jnp.arange(ns)
        return (
            jnp.sum(pair * self.test_diagonal, axis=1)
            + (pair * self.field_diagonal)[own, own]
        )

    def apply(self, hamiltonian: jnp.ndarray, **frequencies: Any) -> jnp.ndarray:
        """Apply the operator to the ``(s, l, m, ...)`` response ``H``."""

        state = jnp.asarray(hamiltonian)
        ns, nl, nm = map(int, state.shape[:3])
        dtype = jnp.real(state).dtype
        # Real and imaginary parts as two real products; XLA otherwise promotes
        # the real factors and does four (1.7x slower in float64 on GPU).
        parts = (
            jnp.stack([state.real, state.imag], -1)
            if jnp.iscomplexobj(state)
            else state
        )
        flat = jnp.swapaxes(parts, 1, 2).reshape(ns, nl * nm, -1)
        flat = jnp.pad(flat, ((0, 0), (0, 1), (0, 0)))
        sonine = jnp.zeros((ns, self.sonine_count + 1, flat.shape[-1]), dtype)
        for rows, cols, matrix in zip(
            self.transform_rows, self.transform_cols, self.transform
        ):
            values = _exact_einsum(
                "brc,sbrp->sbcp", matrix.astype(dtype), flat[:, rows]
            )
            sonine = sonine.at[:, cols].add(values)
        pair = _landau_pair_frequency(ns=ns, dtype=dtype, **frequencies)
        eye = jnp.eye(ns, dtype=dtype)
        mixed = jnp.zeros_like(sonine)
        for index, test, field in zip(
            self.block_index, self.block_test, self.block_field
        ):
            weights = pair[:, :, None, None, None]
            operator = (
                weights * field.astype(dtype)
                + eye[:, :, None, None, None]
                * jnp.sum(weights * test.astype(dtype), axis=1)[:, None]
            )
            mixed = mixed.at[:, index].set(
                _exact_einsum("stbkq,tbqp->sbkp", operator, sonine[:, index])
            )
        result = jnp.zeros_like(flat)
        for rows, cols, matrix in zip(
            self.transform_rows, self.transform_cols, self.transform
        ):
            values = _exact_einsum(
                "brc,sbcp->sbrp", matrix.astype(dtype), mixed[:, cols]
            )
            result = result.at[:, rows].add(values)
        result = jnp.swapaxes(
            result[:, :-1].reshape((ns, nm, nl) + parts.shape[3:]), 1, 2
        )
        if jnp.iscomplexobj(state):
            return jax.lax.complex(result[..., 0], result[..., 1])
        return result


def apply_linearized_landau(
    hamiltonian: jnp.ndarray, blocks: LandauBlocks, **frequencies: Any
) -> jnp.ndarray:
    """Apply the linearized Landau operator (see :class:`LandauBlocks`)."""

    return blocks.apply(hamiltonian, **frequencies)
