"""Physics gate: collisional invariants of every shipped collision operator.

A linearized collision operator must annihilate the collisional invariants. In
the Hermite-Laguerre moment basis with Hermite-major index ``p*(J+1)+j`` these
are left null vectors of the moment matrix ``C``, because the production of a
moment functional ``v`` is ``v^T C N``:

===================  ==================  ==============================
invariant            moment ``(p, j)``   vector
===================  ==================  ==============================
density              ``(0, 0)``          ``e_0``
parallel momentum    ``(1, 0)``          ``e_2``
energy               ``(0, 1)``/``(2,0)``  ``e_1 + e_4/sqrt(2)``
===================  ==================  ==============================

The energy combination is the exact null direction of the drift-kinetic
matrices; the ``1/sqrt(2)`` weight is the perpendicular/parallel split of the
Hermite-Laguerre energy moment.

Two distinct statements are gated here, because they are physically different:

Drift-kinetic operators (``sugama``, ``improved_sugama``, ``coulomb``) act on
the zero-Larmor-radius limit where gyrocenter and particle moments coincide, so
all three invariants must be conserved to machine precision.

The finite-Larmor Coulomb operator acts on *gyrocenter* moments, whose
conservation is modified by gyroaveraging at finite perpendicular wavelength.
The defect must vanish as ``b -> 0`` and enter at first order in
``b = B^2/2``, which is the FLR ordering of the underlying expansion. A defect
growing faster or slower than ``B^2`` would mean the finite-Larmor kernels were
assembled at the wrong order.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from gkx.operators.linear.collision_tables import _finite_wavelength_coulomb_bundle
from gkx.operators.linear.collisions import (
    assemble_drift_kinetic_improved_sugama_matrix,
    assemble_drift_kinetic_sugama_matrix,
    load_collision_moment_matrix,
)

MOMENT_COUNT = 8


def conservation_tolerance() -> float:
    """Return a tolerance matched to the ambient JAX precision.

    The matrices are assembled in the working precision, so the invariant
    production floor is set by cancellation there: ~1e-16 under x64 and ~1e-7
    under the default float32 policy.
    """

    return 1.0e-12 if jnp.zeros(1).dtype == jnp.float64 else 1.0e-5


def coefficient_tolerance() -> float:
    """Absolute tolerance for comparing against published closed forms."""

    return 1.0e-10 if jnp.zeros(1).dtype == jnp.float64 else 1.0e-5


def collisional_invariants() -> dict[str, np.ndarray]:
    """Return the density, parallel-momentum, and energy moment functionals."""

    basis = np.eye(MOMENT_COUNT)
    return {
        "density": basis[0],
        "parallel_momentum": basis[2],
        "energy": basis[1] + basis[4] / np.sqrt(2.0),
    }


def drift_kinetic_matrices() -> dict[str, np.ndarray]:
    """Return the single-species drift-kinetic matrix of each model."""

    density = jnp.asarray([1.0])
    mass = jnp.asarray([1.0])
    temperature = jnp.asarray([1.0])
    return {
        "sugama": np.asarray(
            assemble_drift_kinetic_sugama_matrix(density, mass, temperature)
        )[0, 0],
        "improved_sugama": np.asarray(
            assemble_drift_kinetic_improved_sugama_matrix(density, mass, temperature)
        )[0, 0],
        "coulomb": np.asarray(load_collision_moment_matrix("coulomb")),
    }


def finite_wavelength_matrix(index: int) -> np.ndarray:
    """Return the like-species finite-Larmor matrix at one grid point."""

    arrays, _ = _finite_wavelength_coulomb_bundle()
    return np.asarray(arrays["test_matrix"][index]) + np.asarray(
        arrays["field_matrix"][index]
    )


@pytest.mark.parametrize("model", ["sugama", "improved_sugama", "coulomb"])
def test_drift_kinetic_operators_conserve_all_invariants(model: str) -> None:
    """Density, parallel momentum, and energy are exact drift-kinetic invariants."""

    matrix = drift_kinetic_matrices()[model]
    for name, functional in collisional_invariants().items():
        production = np.abs(functional @ matrix).max()
        assert production < conservation_tolerance(), (
            f"{model} does not conserve {name}: {production:.3e}"
        )


@pytest.mark.parametrize("model", ["sugama", "improved_sugama", "coulomb"])
def test_drift_kinetic_operators_are_dissipative(model: str) -> None:
    """The H-theorem requires a negative-semidefinite symmetrized operator."""

    matrix = drift_kinetic_matrices()[model]
    eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    assert float(eigenvalues.max()) < conservation_tolerance(), model
    # The operator must actually dissipate, not merely fail to grow.
    assert float(eigenvalues.min()) < -0.1, model


SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)
SQRT_1_OVER_PI = np.sqrt(1.0 / np.pi)
SQRT_1_OVER_3PI = np.sqrt(1.0 / (3.0 * np.pi))

# Hermite-major index p*(J+1)+j at J = 1.
MOMENT_INDEX = {(2, 0): 4, (0, 1): 1, (3, 0): 6, (1, 1): 3}

# Frei, Ernst & Ricci (2022), Appendix C, in units of nu_aa. The published
# lists are upper triangular; the transpose follows from self-adjointness.
PUBLISHED_COEFFICIENTS = {
    # Equations (C9a)-(C9f): linearized Coulomb, like species.
    "coulomb": {
        ((2, 0), (2, 0)): -(16 / 15) * SQRT_2_OVER_PI,
        ((2, 0), (0, 1)): -(16 / 15) * SQRT_1_OVER_PI,
        ((0, 1), (0, 1)): -(8 / 15) * SQRT_2_OVER_PI,
        ((3, 0), (3, 0)): -(8 / 5) * SQRT_2_OVER_PI,
        ((3, 0), (1, 1)): -(8 / 5) * SQRT_1_OVER_3PI,
        ((1, 1), (1, 1)): -(28 / 15) * SQRT_2_OVER_PI,
    },
    # Equations (C6a)-(C6f): original Sugama, like species.
    "sugama": {
        ((2, 0), (2, 0)): -(64 / 45) * SQRT_2_OVER_PI,
        ((2, 0), (0, 1)): -(64 / 45) * SQRT_1_OVER_PI,
        ((0, 1), (0, 1)): -(32 / 45) * SQRT_2_OVER_PI,
        ((3, 0), (3, 0)): -(361 / 175) * SQRT_2_OVER_PI,
        ((3, 0), (1, 1)): -(208 / 175) * SQRT_1_OVER_3PI,
        ((1, 1), (1, 1)): -(1187 / 525) * SQRT_2_OVER_PI,
    },
}


@pytest.mark.parametrize("model", ["coulomb", "sugama"])
def test_matrices_match_published_closed_form_coefficients(model: str) -> None:
    """Assert the shipped tables against the published closed forms.

    This is the strongest available check on the generated coefficients: every
    retained entry has an exact analytic value in Frei, Ernst & Ricci (2022),
    Appendix C, so agreement is a statement about the numbers themselves rather
    than about an internal consistency relation.

    GKX stores the opposite Laguerre sign convention to the paper
    (``laguerre_convention: gkx_opposite_to_paper``), so a published entry maps
    onto the stored one through ``(-1)^(j + j')``. That flips exactly the
    couplings between different Laguerre parities and leaves same-parity
    entries alone, which is what makes this a convention check as well as a
    value check.
    """

    matrix = drift_kinetic_matrices()[model]
    for (left, right), published in PUBLISHED_COEFFICIENTS[model].items():
        convention = (-1.0) ** (left[1] + right[1])
        expected = convention * published
        for row, column in ((left, right), (right, left)):
            stored = float(matrix[MOMENT_INDEX[row], MOMENT_INDEX[column]])
            assert stored == pytest.approx(expected, abs=coefficient_tolerance()), (
                f"{model} C[{row},{column}]: stored {stored:+.10f}, "
                f"published {published:+.10f} in the paper convention"
            )


def test_sugama_and_coulomb_share_the_published_temperature_block_ratio() -> None:
    """The Sugama (2,0)/(0,1) block is exactly 4/3 of the Coulomb block.

    Frei, Ernst & Ricci (2022) note this relation, and it does not extend to
    the heat-flux block, so it separates the two models structurally rather
    than by an overall scale.
    """

    coulomb = drift_kinetic_matrices()["coulomb"]
    sugama = drift_kinetic_matrices()["sugama"]
    for pair in (((2, 0), (2, 0)), ((2, 0), (0, 1)), ((0, 1), (0, 1))):
        row, column = MOMENT_INDEX[pair[0]], MOMENT_INDEX[pair[1]]
        assert float(sugama[row, column]) == pytest.approx(
            (4.0 / 3.0) * float(coulomb[row, column]), rel=1.0e-5
        )

    # The heat-flux block deviates, so the models are not a rescaling.
    heat = (MOMENT_INDEX[(3, 0)], MOMENT_INDEX[(3, 0)])
    ratio = float(sugama[heat]) / float(coulomb[heat])
    assert ratio == pytest.approx(361.0 / 280.0, rel=1.0e-5)


@pytest.mark.parametrize("model", ["sugama", "improved_sugama", "coulomb"])
def test_drift_kinetic_operators_are_self_adjoint(model: str) -> None:
    """The linearized operator is self-adjoint in the Maxwellian-weighted basis.

    Onsager symmetry is an independent structural check: it constrains the
    off-diagonal moment couplings, which conservation and dissipativity alone
    do not. In the orthonormal Hermite-Laguerre basis it makes the drift-kinetic
    matrix exactly symmetric.
    """

    matrix = drift_kinetic_matrices()[model]
    asymmetry = np.abs(matrix - matrix.T).max() / np.abs(matrix).max()
    assert asymmetry < conservation_tolerance(), f"{model}: {asymmetry:.3e}"


def test_finite_larmor_self_adjointness_breaks_at_first_order_in_b() -> None:
    """Gyroaveraging breaks plain symmetry at first order in b, and no faster.

    At finite perpendicular wavelength the operator is self-adjoint with
    respect to a gyroaveraging-weighted inner product rather than the plain
    one, so the stored matrix acquires an antisymmetric part. That part must
    vanish at b = 0 and grow as B^2, matching the conservation defect.
    """

    _, metadata = _finite_wavelength_coulomb_bundle()
    grid = np.asarray(metadata["bessel_argument_grid"], dtype=float)

    zero = finite_wavelength_matrix(0)
    assert (
        np.abs(zero - zero.T).max() / np.abs(zero).max() < conservation_tolerance()
    ), "the drift-kinetic limit must stay self-adjoint"

    small = (grid > 0.0) & (grid <= 0.5)
    asymmetry = np.array(
        [
            np.abs(
                (matrix := finite_wavelength_matrix(index)) - matrix.T
            ).max()
            for index in np.flatnonzero(small)
        ]
    )
    assert np.all(asymmetry > 0.0)
    exponent = float(np.polyfit(np.log(grid[small]), np.log(asymmetry), 1)[0])
    assert 1.8 <= exponent <= 2.3, f"asymmetry scales as B^{exponent:.3f}, not B^2"


def test_finite_larmor_coulomb_conserves_invariants_at_zero_wavelength() -> None:
    """At b = 0 the gyrocenter and particle moments coincide, so conservation is exact."""

    _, metadata = _finite_wavelength_coulomb_bundle()
    assert metadata["bessel_argument_grid"][0] == 0.0
    matrix = finite_wavelength_matrix(0)
    for name, functional in collisional_invariants().items():
        production = np.abs(functional @ matrix).max()
        assert production < conservation_tolerance(), (
            f"finite-Larmor b=0 breaks {name}: {production:.3e}"
        )


def test_finite_larmor_conservation_defect_is_first_order_in_b() -> None:
    """The gyrocenter conservation defect must enter at first order in b = B^2/2.

    This is the sharpest available check that the finite-Larmor kernels carry
    the right order: a defect scaling as B^2 is linear in b, while a wrong
    kernel assembly would show B^1 or B^4.
    """

    _, metadata = _finite_wavelength_coulomb_bundle()
    grid = np.asarray(metadata["bessel_argument_grid"], dtype=float)
    small = (grid > 0.0) & (grid <= 0.5)
    assert small.sum() >= 3, "need several small-B points to fit an exponent"

    for name, functional in collisional_invariants().items():
        defect = np.array(
            [
                np.abs(functional @ finite_wavelength_matrix(index)).max()
                for index in np.flatnonzero(small)
            ]
        )
        assert np.all(defect > 0.0), f"{name} defect vanishes identically at finite b"
        exponent = float(
            np.polyfit(np.log(grid[small]), np.log(defect), 1)[0]
        )
        assert 1.8 <= exponent <= 2.2, f"{name} defect scales as B^{exponent:.3f}, not B^2"

    # Monotone growth with wavelength: FLR corrections do not fortuitously cancel.
    density = collisional_invariants()["density"]
    defects = [
        np.abs(density @ finite_wavelength_matrix(index)).max()
        for index in range(int(small.sum()) + 1)
    ]
    assert np.all(np.diff(defects) > 0.0)


def test_laguerre_transform_is_well_conditioned_at_high_resolution() -> None:
    """The velocity transform must stay accurate at the resolutions physics needs.

    Storing unweighted Laguerre polynomials makes ``to_grid`` reach 1e14 by
    nl = 16 and 1e19 by nl = 20, because the largest Gauss node grows like
    4*nj, and the separately applied exp(-x) weight then has to cancel a
    growing number of digits. Folding exp(-x/2) into the recurrence bounds the
    stored values by the Szego bound instead, so the round-trip identity stays
    near machine precision.

    Published convergence studies ask for up to J = 16 Laguerre moments, and
    the collisionless zonal-flow residual needs more, so this range has to hold
    with headroom rather than sit at the edge of a cliff.
    """

    from gkx.core.velocity import laguerre_transform

    for resolution in (8, 16, 20, 24, 32, 64, 96):
        to_grid, to_spectral, _ = laguerre_transform(resolution)
        to_grid = np.asarray(to_grid)
        identity = np.abs(
            to_grid @ np.asarray(to_spectral) - np.eye(resolution)
        ).max()
        assert identity < 1.0e-8, f"nl={resolution} round-trip {identity:.3e}"
        # The Szego bound is what keeps the transform conditioned.
        assert np.abs(to_grid).max() <= 1.0 + 1.0e-9, (
            f"nl={resolution} stores unweighted polynomials again: "
            f"max|to_grid| = {np.abs(to_grid).max():.3e}"
        )


def test_finite_larmor_tables_ship_multiple_resolutions() -> None:
    """Each shipped resolution must load, verify, and satisfy the same physics.

    Published convergence studies need well past the eight moments that match
    the drift-kinetic tables, so more than one resolution has to be available
    and every one of them has to pass the same structural checks.
    """

    from gkx.operators.linear.collision_tables import (
        FINITE_WAVELENGTH_MOMENT_COUNTS,
        _finite_wavelength_coulomb_bundle,
        build_finite_wavelength_coulomb_operator,
        finite_wavelength_coulomb_metadata,
    )

    assert len(FINITE_WAVELENGTH_MOMENT_COUNTS) >= 2

    for moments in FINITE_WAVELENGTH_MOMENT_COUNTS:
        metadata = finite_wavelength_coulomb_metadata(moments)
        hermite = int(metadata["maximum_hermite_order"])
        laguerre = int(metadata["maximum_laguerre_order"])
        assert (hermite + 1) * (laguerre + 1) == moments

        arrays, _ = _finite_wavelength_coulomb_bundle(moments)
        matrix = np.asarray(arrays["test_matrix"][0]) + np.asarray(
            arrays["field_matrix"][0]
        )
        assert matrix.shape == (moments, moments)

        # Same drift-kinetic limit physics at every resolution: the invariants
        # are conserved and the operator is self-adjoint and dissipative.
        basis = np.eye(moments)
        stride = laguerre + 1
        invariants = (
            basis[0],
            basis[stride],
            basis[1] + basis[2 * stride] / np.sqrt(2.0),
        )
        for functional in invariants:
            assert np.abs(functional @ matrix).max() < conservation_tolerance()
        assert (
            np.abs(matrix - matrix.T).max() / np.abs(matrix).max()
            < conservation_tolerance()
        )
        assert float(np.linalg.eigvalsh(0.5 * (matrix + matrix.T)).max()) < 1.0e-9

        operator = build_finite_wavelength_coulomb_operator(
            jnp.asarray([1.0]), jnp.asarray([1.0]), jnp.asarray([1.0]), moments
        )
        assert operator.test_table.shape[-1] == moments

    with pytest.raises(ValueError, match="no shipped finite-wavelength"):
        _finite_wavelength_coulomb_bundle(12)
