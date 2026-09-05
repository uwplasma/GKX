"""Physics gates: the shipped collision operators.

Everything here is a statement about the Hermite-Laguerre collision operators
themselves -- their exact invariants and published coefficients, their
reduction limits, the transport coefficient they must reproduce, and the cost
envelope of the finite-Larmor tables that carry them. Each block below keeps
the module docstring of the file it came from, because those docstrings record
the provenance of the reference values being asserted.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gkx.config import CycloneBaseCase, GridConfig
from gkx.core_grid import build_spectral_grid
from gkx.geometry import SAlphaGeometry
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.collision_tables import _finite_wavelength_coulomb_bundle
from gkx.operators.linear.collisions import (
    assemble_drift_kinetic_improved_sugama_matrix,
    assemble_drift_kinetic_sugama_matrix,
    load_collision_moment_matrix,
    solve_driven_collision_response,
)
from gkx.operators.linear.params import LinearParams
from gkx.operators.linear.rhs import linear_rhs_cached
from gkx.solvers_time_runners import _resolve_config_collision_operator

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "artifacts"))

from build_linear_validation_artifacts import (  # noqa: E402
    build_finite_wavelength_coulomb_pair_tables,
    coulomb_drift_kinetic_moment_matrices,
    like_species_test_particle_gram_matrices,
    like_species_test_particle_polarization,
)


MOMENT_COUNT = 8


@pytest.mark.parametrize("pmax,jmax", [(3, 1), (5, 2), (7, 3)])
def test_like_species_test_particle_dirichlet_quadrature(pmax, jmax):
    c0, d = like_species_test_particle_gram_matrices(pmax, jmax)
    refined = like_species_test_particle_gram_matrices(
        pmax, jmax, radial_nodes=192, pitch_nodes=64
    )
    np.testing.assert_allclose((c0, d), refined, rtol=1e-11, atol=1e-12)
    np.testing.assert_array_equal(c0[:, 0], 0.0)
    assert np.linalg.eigvalsh(c0)[-1] <= 1e-12
    assert np.linalg.eigvalsh(d)[0] > 0.0
    if pmax < 7:
        exact_dk = coulomb_drift_kinetic_moment_matrices(
            pmax, jmax, 1.0, 1.0, digits=40
        )[0]
        np.testing.assert_allclose(c0, exact_dk, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize(
    "p,j,nr,nxi", [(-1, 0, 48, 32), (0, -1, 48, 32), (0, 0, 0, 32), (0, 0, 48, 0)]
)
def test_test_particle_quadrature_rejects_invalid_orders(p, j, nr, nxi):
    with pytest.raises(ValueError, match="orders"):
        like_species_test_particle_gram_matrices(p, j, radial_nodes=nr, pitch_nodes=nxi)


@pytest.mark.parametrize("pmax,jmax", [(3, 1), (5, 2)])
@pytest.mark.parametrize("b", [0.0, 1.0, 4.0])
def test_test_particle_polarization_quadrature_and_j0_source(pmax, jmax, b):
    from scipy.special import gammaln

    result = like_species_test_particle_polarization(pmax, jmax, b)
    refined = like_species_test_particle_polarization(
        pmax, jmax, b, radial_nodes=192, pitch_nodes=64
    )
    np.testing.assert_allclose(result, refined, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(result.reshape(pmax + 1, jmax + 1)[1::2], 0, atol=1e-14)
    if b == 0:
        np.testing.assert_array_equal(result, 0.0)
        return
    # J0(B sqrt(x)) = sum_j exp(-B²/4)(B²/4)^j/j! L_j(x).
    # Resolve the source independently of the retained output moments.
    c0, d = like_species_test_particle_gram_matrices(pmax, 24)
    j = np.arange(25)
    source = np.zeros((pmax + 1, 25))
    source[0] = np.exp(-b * b / 4 + j * np.log(b * b / 4) - gammaln(j + 1))
    projected = ((c0 - b * b * d) @ source.ravel()).reshape(pmax + 1, 25)
    np.testing.assert_allclose(projected[:, : jmax + 1].ravel(), result, atol=1e-11)


def test_test_particle_polarization_quadratic_limit():
    c0, d = like_species_test_particle_gram_matrices(3, 1)
    b = 1e-4
    result = like_species_test_particle_polarization(3, 1, b)
    # J0 = 1 - B²*x/4 + O(B⁴), x = L0-L1, C0[:,0] = 0.
    np.testing.assert_allclose(
        result / b**2, c0[:, 1] / 4 - d[:, 0], rtol=2e-8, atol=1e-10
    )


@pytest.mark.parametrize("b", [-1.0, np.nan, np.inf])
def test_test_particle_polarization_rejects_invalid_wavelength(b):
    with pytest.raises(ValueError, match="bessel_argument"):
        like_species_test_particle_polarization(3, 1, b)


@pytest.mark.parametrize("error", [0.0, 1e-5, np.nan, np.inf])
@pytest.mark.parametrize("component", [0, 1])
def test_collision_table_check_precedes_publication(
    monkeypatch, tmp_path, error, component
):
    import build_finite_wavelength_coulomb_data as generator
    import build_linear_validation_artifacts as reference

    count = len(generator.BESSEL_ARGUMENTS)
    blocks = [np.zeros((count, count, 8, 8)) for _ in range(2)]
    blocks += [np.zeros((count, count, 8)) for _ in range(4)]
    blocks[component][0, 0, 0, 0] = error
    monkeypatch.setattr(generator, "build_tables", lambda *args: tuple(blocks))
    monkeypatch.setattr(
        reference,
        "coulomb_drift_kinetic_moment_matrices",
        lambda *args, **kwargs: (np.zeros((8, 8)),) * 2,
    )
    monkeypatch.setattr(generator, "DATA_DIR", tmp_path)
    monkeypatch.setattr(generator, "REPO_ROOT", tmp_path)
    paths = [tmp_path / f"{generator.STEM}.{suffix}" for suffix in ("npz", "json")]
    for path in paths:
        path.write_bytes(b"existing table")
    assert generator.main(["--check"]) == (0 if error == 0 else 1)
    for path in paths:
        assert (path.read_bytes() == b"existing table") == (error != 0)


@pytest.mark.parametrize("component", range(6))
@pytest.mark.parametrize("error", [np.nan, np.inf])
def test_collision_table_publication_rejects_nonfinite(
    monkeypatch, tmp_path, component, error
):
    import build_finite_wavelength_coulomb_data as generator

    blocks = {name: np.zeros(2) for name in generator.BLOCK_NAMES}
    blocks[generator.BLOCK_NAMES[component]][-1] = error
    monkeypatch.setattr(generator, "DATA_DIR", tmp_path)
    paths = [tmp_path / f"{generator.STEM}.{suffix}" for suffix in ("npz", "json")]
    for path in paths:
        path.write_bytes(b"existing table")
    with pytest.raises(ValueError, match="non-finite"):
        generator.write_artifacts(blocks, digits=40)
    assert all(path.read_bytes() == b"existing table" for path in paths)


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


MOMENT_INDEX = {(2, 0): 4, (0, 1): 1, (3, 0): 6, (1, 1): 3}


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
    """Regress stored-table asymmetry scaling; not a self-adjointness proof.

    A physical metric and the g/h/field mapping must be derived independently
    before interpreting this measured defect as an admissible property.
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
            np.abs((matrix := finite_wavelength_matrix(index)) - matrix.T).max()
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

    This regresses the stored tables, not finite-k conservation: a wrong
    assembly can also have a B^2 defect. Physical moment functionals and
    field-response terms require independent verification.
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
        exponent = float(np.polyfit(np.log(grid[small]), np.log(defect), 1)[0])
        assert 1.8 <= exponent <= 2.2, (
            f"{name} defect scales as B^{exponent:.3f}, not B^2"
        )

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

    from gkx.core_velocity import laguerre_transform

    for resolution in (8, 16, 20, 24, 32, 64, 96):
        to_grid, to_spectral, _ = laguerre_transform(resolution)
        to_grid = np.asarray(to_grid)
        identity = np.abs(to_grid @ np.asarray(to_spectral) - np.eye(resolution)).max()
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


# ---- from test_multispecies_coulomb_reduction.py ----
# Physics gate: the multispecies finite-Larmor Coulomb operator reduces to the
# drift-kinetic multispecies Coulomb operator as the Bessel argument ``b -> 0``.
#
# This is the foundational reduction limit for the multispecies gyrokinetic
# Coulomb collision operator. The finite-wavelength pair generation
# (``build_finite_wavelength_coulomb_pair_tables``) accepts an arbitrary
# mass ratio ``sigma`` and temperature ratio ``tau`` (Frei, Ball, Hoffmann,
# Jorge, Ricci & Stenger 2021, arXiv:2104.11480, Eqs. 3.47-3.50). As
# ``b_a = k_perp v_{th,a}/Omega_a -> 0`` it must recover the drift-kinetic
# multispecies Coulomb operator of Jorge et al. (2018), which GKX generates via
# ``coulomb_drift_kinetic_moment_matrices`` (arXiv:2104.11480, Eqs. 3.55-3.56).
#
# The finite-wavelength tables use the signed-Laguerre runtime convention, so the
# comparison applies the ``(-1)^lag (x) (-1)^lag`` sign transform to the
# finite-wavelength blocks before matching. Verified for like-species,
# electron-ion, and arbitrary unequal pairs -- closing the "unequal-species
# finite-wavelength Coulomb is unvalidated" gap.


def _signed_laguerre_convention(hermite: int, laguerre: int) -> np.ndarray:
    """Sign transform between the finite-wavelength and drift-kinetic bases."""

    sign = np.asarray(
        [(-1.0) ** lag for _h in range(hermite + 1) for lag in range(laguerre + 1)]
    )
    return sign[:, None] * sign[None, :]


@pytest.mark.parametrize(
    ("mass_ratio", "temperature_ratio", "label"),
    [
        (1.0, 1.0, "like-species"),
        (1836.0, 1.0, "electron-ion"),
        (0.5, 2.0, "arbitrary-unequal"),
    ],
)
def test_finite_wavelength_coulomb_reduces_to_drift_kinetic_at_b0(
    mass_ratio: float, temperature_ratio: float, label: str
) -> None:
    hermite, laguerre, digits = 1, 1, 40
    convention = _signed_laguerre_convention(hermite, laguerre)

    dk_test, dk_field = (
        np.asarray(matrix, dtype=float)
        for matrix in coulomb_drift_kinetic_moment_matrices(
            hermite, laguerre, mass_ratio, temperature_ratio, digits=digits
        )[:2]
    )

    # Two tiny, strictly increasing Bessel arguments; the (target=source=0)
    # block is the b -> 0 endpoint of the finite-wavelength pair table.
    tables = build_finite_wavelength_coulomb_pair_tables(
        (1.0e-4, 2.0e-4),
        hermite,
        laguerre,
        mass_ratio=mass_ratio,
        temperature_ratio=temperature_ratio,
        digits=digits,
    )
    fw_test = np.asarray(tables[0], dtype=float)[0, 0]
    fw_field = np.asarray(tables[1], dtype=float)[0, 0]

    # b -> 0 reduction to the drift-kinetic multispecies Coulomb operator, in
    # the shared (unsigned-Laguerre) convention, for both test and field parts.
    np.testing.assert_allclose(
        fw_test * convention, dk_test, atol=1.0e-6, err_msg=f"{label}: test part"
    )
    np.testing.assert_allclose(
        fw_field * convention, dk_field, atol=1.0e-6, err_msg=f"{label}: field part"
    )

    # Density (moment 0) is an exact collisional invariant: no production row.
    np.testing.assert_allclose(dk_test[0, :], 0.0, atol=1.0e-9)
    np.testing.assert_allclose(dk_field[0, :], 0.0, atol=1.0e-9)


# ---- from test_spitzer_conductivity.py ----
# Physics gate: the Coulomb operator reproduces the Spitzer-Harm conductivity.
#
# The stationary Spitzer problem drives a uniform parallel electric field against
# electron collisions and asks for the resulting current. It is the cheapest
# end-to-end test of a collision operator that has a closed-form answer, and it
# exercises the unlike-species (electron-ion) coefficients that the like-species
# conservation gates cannot reach.
#
# Electrons collide with themselves and with a fixed Maxwellian ion background of
# charge ``Z``. Writing the steady moment balance as ``C N + s = 0``, with ``s``
# the linearized parallel-field drive that survives only in ``(p, j) = (1, 0)``,
# the parallel flow follows from ``u_e = N^{10} v_Te / sqrt(2)``.
#
# Quasineutrality ``n_i Z = n_e`` makes the electron-ion collision frequency scale
# as ``nu_ei ~ n_i Z^2 = n_e Z``, so the total operator is
# ``C_ee^T + C_ee^F + Z C_ei^T`` in units of ``nu_ee``.
#
# The absolute conductivity depends on the normalization convention, of which
# three incompatible ones appear in this literature, so this gate asserts the
# *ratio*
#
#     gamma_E(Z) = sigma(e-e and e-i) / sigma(e-i only)
#
# which is convention free. It is the classic Spitzer-Harm correction to the
# Lorentz-gas conductivity, tabulated in Spitzer & Harm, *Phys. Rev.* 89, 977
# (1953), and it must approach unity as ``Z -> infinity`` because the
# electron-ion term then dominates.


SPITZER_HARM_GAMMA_E = {1: 0.5816, 2: 0.6833, 4: 0.7849, 16: 0.9225}


HERMITE_ORDER = 7


LAGUERRE_ORDER = 2


DIGITS = 40


def collision_blocks(
    hermite: int = HERMITE_ORDER, laguerre: int = LAGUERRE_ORDER
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the electron self-collision and electron-ion test matrices."""

    self_test, self_field = (
        np.asarray(matrix, dtype=float)
        for matrix in coulomb_drift_kinetic_moment_matrices(
            hermite, laguerre, 1.0, 1.0, digits=DIGITS
        )[:2]
    )
    # A fixed Maxwellian ion background is the m_e/m_i -> 0 limit.
    ion_test = np.asarray(
        coulomb_drift_kinetic_moment_matrices(
            hermite, laguerre, 1.0e-12, 1.0, digits=DIGITS
        )[0],
        dtype=float,
    )
    return self_test, self_field, ion_test


def driven_parallel_flow(
    blocks: tuple[np.ndarray, np.ndarray, np.ndarray],
    laguerre: int,
    charge: float,
    *,
    include_self_collisions: bool,
) -> float:
    """Solve the stationary Spitzer problem and return the parallel flow."""

    self_test, self_field, ion_test = blocks
    matrix = charge * ion_test
    if include_self_collisions:
        matrix = matrix + self_test + self_field

    size = matrix.shape[0]
    momentum_index = 1 * (laguerre + 1)
    source = np.zeros(size)
    source[momentum_index] = np.sqrt(2.0)

    # Electron-ion collisions break parallel-momentum conservation, so density
    # is the only exact invariant of the total operator and the only mode that
    # has to be projected out.
    active = tuple(index for index in range(size) if index != 0)
    moments = np.asarray(
        solve_driven_collision_response(
            jnp.asarray(matrix), jnp.asarray(-source), active_modes=active
        )
    )
    return float(moments[momentum_index] / np.sqrt(2.0))


def spitzer_harm_ratio(
    blocks: tuple[np.ndarray, np.ndarray, np.ndarray], laguerre: int, charge: float
) -> float:
    """Return gamma_E = sigma / sigma_Lorentz at the given ion charge."""

    with_self = driven_parallel_flow(
        blocks, laguerre, charge, include_self_collisions=True
    )
    lorentz = driven_parallel_flow(
        blocks, laguerre, charge, include_self_collisions=False
    )
    return with_self / lorentz


@pytest.fixture(scope="module")
def blocks() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return collision_blocks()


def test_spitzer_harm_conductivity_ratio(blocks) -> None:
    """gamma_E(Z) must match the tabulated Spitzer-Harm values."""

    for charge, published in SPITZER_HARM_GAMMA_E.items():
        ratio = spitzer_harm_ratio(blocks, LAGUERRE_ORDER, float(charge))
        relative = abs(ratio - published) / published
        assert relative < 0.015, (
            f"Z={charge}: gamma_E = {ratio:.4f}, Spitzer-Harm {published:.4f}, "
            f"{relative * 100:.2f}% away"
        )


def test_spitzer_ratio_approaches_the_lorentz_limit(blocks) -> None:
    """As Z grows the electron-ion term dominates and gamma_E -> 1."""

    ratios = [
        spitzer_harm_ratio(blocks, LAGUERRE_ORDER, charge)
        for charge in (1.0, 2.0, 4.0, 16.0, 100.0, 1000.0)
    ]
    # Monotone approach from below: electron-electron collisions can only
    # reduce the current relative to the Lorentz gas.
    assert all(0.0 < ratio < 1.0 for ratio in ratios)
    assert all(np.diff(ratios) > 0.0)
    assert ratios[-1] > 0.99, f"gamma_E(Z=1000) = {ratios[-1]:.4f} should approach 1"


def test_spitzer_ratio_converges_with_moment_number() -> None:
    """The ratio must be converged in the Hermite-Laguerre truncation.

    A result that still moves with resolution would not be evidence about the
    operator, only about the truncation.
    """

    coarse = collision_blocks(5, 2)
    fine = collision_blocks(HERMITE_ORDER, LAGUERRE_ORDER)
    for charge in (1.0, 4.0):
        coarse_ratio = spitzer_harm_ratio(coarse, 2, charge)
        fine_ratio = spitzer_harm_ratio(fine, LAGUERRE_ORDER, charge)
        drift = abs(fine_ratio - coarse_ratio) / fine_ratio
        assert drift < 0.01, (
            f"Z={charge}: gamma_E moved {drift * 100:.2f}% between "
            f"(5,2) and ({HERMITE_ORDER},{LAGUERRE_ORDER})"
        )


# ---- from test_collision_operator_cost.py ----
# Performance guard: cost of the collision operators in the linear RHS.
#
# The finite-Larmor Coulomb operator interpolates its tables at every grid point,
# so the compiler sees a distinct moment matrix per ``(ky, kx, z)``. That storage
# grows as ``n^2`` in the moment count while the state grows as ``n``, which is
# what eventually bounds the reachable resolution:
#
# ===================  ==========  ==========================================
# moments ``n``        grid        per-point matrix storage
# ===================  ==========  ==========================================
# 8   ``(4, 2)``       32x64x32    0.07 GB
# 18  ``(6, 3)``       32x64x32    0.34 GB
# 128 ``(16, 8)``      32x64x32    17 GB, past a 16 GB card
# 512 ``(32, 16)``     32x64x32    275 GB
# ===================  ==========  ==========================================
#
# Published convergence studies ask for ``(16, 8)`` for linear Cyclone-base-case
# ITG and ``(32, 16)`` converged, so this is the mechanism that has to change
# before those resolutions are reachable. It is comfortable at the resolutions
# GKX ships today, which is why this file measures and bounds the cost rather
# than asserting a target that the current implementation cannot meet.
#
# The bounds are ratios against the built-in diagonal operator rather than
# absolute byte counts, so they track the structural cost and tolerate compiler
# and library changes.


GRID = GridConfig(Nx=8, Ny=16, Nz=32, Lx=62.8, Ly=62.8)


def compiled_rhs_cost(collision_operator: str, hermite: int, laguerre: int):
    """Return (flops, bytes, temp_bytes) for one compiled linear RHS."""

    config = CycloneBaseCase(grid=GRID)
    grid = build_spectral_grid(config.grid)
    geometry = SAlphaGeometry.from_config(config.geometry)
    parameters = LinearParams(nu=0.05)
    state = jnp.zeros(
        (hermite, laguerre, grid.ky.size, grid.kx.size, grid.z.size),
        dtype=jnp.complex128,
    )
    cache = build_linear_cache(grid, geometry, parameters, hermite, laguerre)
    time_config = dataclasses.replace(
        config.time, collision_operator=collision_operator
    )
    operator = _resolve_config_collision_operator(time_config, parameters, state)

    def rhs(value):
        return linear_rhs_cached(
            value, cache, parameters, use_jit=False, collision_operator=operator
        )[0]

    compiled = jax.jit(rhs).lower(state).compile()
    analysis = compiled.cost_analysis()
    return (
        float(analysis["flops"]),
        float(analysis["bytes accessed"]),
        float(compiled.memory_analysis().temp_size_in_bytes),
    )


@pytest.mark.parametrize(
    ("hermite", "laguerre", "temp_ratio_bound"),
    [(4, 2, 12.0), (6, 3, 24.0)],
)
def test_finite_larmor_overhead_stays_within_its_measured_envelope(
    hermite: int, laguerre: int, temp_ratio_bound: float
) -> None:
    """The finite-Larmor operator's extra temporary storage must not blow up.

    Measured on this grid: 5.9x the diagonal operator at 8 moments and 12.1x at
    18. The bounds are set at roughly twice those so ordinary compiler drift
    passes while a structural regression, such as losing a fusion or
    materializing the tables for every species pair, fails.
    """

    _, _, diagonal_temp = compiled_rhs_cost("lenard_bernstein", hermite, laguerre)
    _, _, coulomb_temp = compiled_rhs_cost("coulomb_finite_kperp", hermite, laguerre)

    assert diagonal_temp > 0.0
    ratio = coulomb_temp / diagonal_temp
    assert ratio < temp_ratio_bound, (
        f"({hermite},{laguerre}): finite-Larmor temporaries are {ratio:.1f}x the "
        f"diagonal operator, above the {temp_ratio_bound}x envelope"
    )


def test_finite_larmor_cost_grows_no_faster_than_the_moment_count_squared() -> None:
    """Cost must track the n^2 matrix, not something steeper.

    Going from 8 to 18 moments raises n^2 by 5.06x. A growth rate materially
    above that would mean the implementation had acquired an extra factor, for
    instance interpolating per species pair or per Runge-Kutta stage.
    """

    _, _, small = compiled_rhs_cost("coulomb_finite_kperp", 4, 2)
    _, _, large = compiled_rhs_cost("coulomb_finite_kperp", 6, 3)
    moment_ratio = (18 / 8) ** 2
    growth = large / small
    assert growth < 1.5 * moment_ratio, (
        f"temporaries grew {growth:.2f}x for an n^2 ratio of {moment_ratio:.2f}x"
    )


def test_diagonal_operator_stays_cheap() -> None:
    """The built-in operator must not acquire per-point matrix storage.

    It is the fallback for every path that cannot carry a moment operator, so a
    regression here would be felt everywhere.
    """

    _, _, small = compiled_rhs_cost("lenard_bernstein", 4, 2)
    _, _, large = compiled_rhs_cost("lenard_bernstein", 6, 3)
    # Diagonal damping stores O(n) per point, so cost should track the state
    # size (2.25x from 8 to 18 moments), not n^2.
    assert large / small < 4.0, f"diagonal operator grew {large / small:.2f}x"
