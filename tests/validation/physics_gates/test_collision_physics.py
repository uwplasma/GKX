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
from gkx.config import GeometryConfig
from gkx.core_grid import select_ky_grid
from gkx.operators.linear.params import LinearTerms, linear_params_for_geometry
from gkx.solvers_linear_integrators import integrate_linear_diagnostics
import math
import matplotlib.pyplot as plt
from PIL import Image
from gkx.artifacts.figure_style import save_figure
from gkx.geometry.flux_tube import sample_flux_tube_geometry
from gkx.objectives.core import (
    _default_gradient_linear_params,
    solver_objective_vector_from_geometry,
)
from gkx.operators.linear.streaming import abs_z_periodic
from gkx.terms.linear_terms import (
    hermite_closure_coefficient,
    linked_streaming_contribution,
)
from scripts.artifacts.build_landau_damping_figure import (
    _NU_SCAN,
    evolve,
    exact_root,
    fit_standing_wave,
    operator_matrix,
)
import json
from gkx.diagnostics.analysis import (
    BranchContinuationMetrics,
    LateTimeLinearMetrics,
    NonlinearHeatFluxConvergenceMetrics,
    NonlinearWindowMetrics,
    ObservedOrderMetrics,
)
from gkx.diagnostics.modes import EigenfunctionComparisonMetrics
import scripts.checks._gates.validation_gates as validation_gates
from scripts.checks._gates.validation_gates import (
    GateReport,
    ScalarGateResult,
    ZonalFlowResponseMetrics,
    branch_continuity_gate_report,
    eigenfunction_gate_report,
    evaluate_scalar_gate,
    gate_report,
    gate_report_to_dict,
    linear_metrics_gate_report,
    nonlinear_heat_flux_convergence_gate_report,
    nonlinear_window_gate_report,
    observed_order_gate_report,
    zonal_response_gate_report,
)
from scripts.checks._gates.zonal_validation import (
    zonal_flow_response_metrics,
)


sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "artifacts"))

from build_linear_validation_artifacts import (  # noqa: E402
    build_finite_wavelength_coulomb_pair_tables,
    coulomb_drift_kinetic_moment_matrices,
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
# ``(Nl, Nm)``
# ===================  ==========  ==========================================
# 8   ``(2, 4)``       32x64x32    0.07 GB
# 18  ``(3, 6)``       32x64x32    0.34 GB
# 128 ``(8, 16)``      32x64x32    17 GB, past a 16 GB card
# 512 ``(16, 32)``     32x64x32    275 GB
# ===================  ==========  ==========================================
#
# Published convergence studies ask for 16 Hermite by 8 Laguerre moments for
# linear Cyclone-base-case ITG and 32 by 16 converged, so this is the mechanism that has to change
# before those resolutions are reachable. It is comfortable at the resolutions
# GKX ships today, which is why this file measures and bounds the cost rather
# than asserting a target that the current implementation cannot meet.
#
# The bounds are ratios against the built-in diagonal operator rather than
# absolute byte counts, so they track the structural cost and tolerate compiler
# and library changes.


GRID = GridConfig(Nx=8, Ny=16, Nz=32, Lx=62.8, Ly=62.8)


def compiled_rhs_cost(collision_operator: str, nl: int, nm: int):
    """Return (flops, bytes, temp_bytes) for one compiled RHS at ``(Nl, Nm)``."""

    config = CycloneBaseCase(grid=GRID)
    grid = build_spectral_grid(config.grid)
    geometry = SAlphaGeometry.from_config(config.geometry)
    parameters = LinearParams(nu=0.05)
    state = jnp.zeros(
        (nl, nm, grid.ky.size, grid.kx.size, grid.z.size),
        dtype=jnp.complex128,
    )
    cache = build_linear_cache(grid, geometry, parameters, nl, nm)
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
    ("nl", "nm", "temp_ratio_bound"),
    [(2, 4, 12.0), (3, 6, 24.0)],
)
def test_finite_larmor_overhead_stays_within_its_measured_envelope(
    nl: int, nm: int, temp_ratio_bound: float
) -> None:
    """The finite-Larmor operator's extra temporary storage must not blow up.

    Measured on this grid (jax 0.10.2, CPU): 4.6x the diagonal operator at
    (Nl, Nm) = (2, 4) and 9.6x at (3, 6). The bounds are set at roughly two and
    a half times those so ordinary compiler drift
    passes while a structural regression, such as losing a fusion or
    materializing the tables for every species pair, fails.
    """

    _, _, diagonal_temp = compiled_rhs_cost("lenard_bernstein", nl, nm)
    _, _, coulomb_temp = compiled_rhs_cost("coulomb_finite_kperp", nl, nm)

    assert diagonal_temp > 0.0
    ratio = coulomb_temp / diagonal_temp
    assert ratio < temp_ratio_bound, (
        f"({nl},{nm}): finite-Larmor temporaries are {ratio:.1f}x the "
        f"diagonal operator, above the {temp_ratio_bound}x envelope"
    )


def test_finite_larmor_cost_grows_no_faster_than_the_moment_count_squared() -> None:
    """Cost must track the n^2 matrix, not something steeper.

    Going from 8 to 18 moments raises n^2 by 5.06x. A growth rate materially
    above that would mean the implementation had acquired an extra factor, for
    instance interpolating per species pair or per Runge-Kutta stage.
    """

    _, _, small = compiled_rhs_cost("coulomb_finite_kperp", 2, 4)
    _, _, large = compiled_rhs_cost("coulomb_finite_kperp", 3, 6)
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

    _, _, small = compiled_rhs_cost("lenard_bernstein", 2, 4)
    _, _, large = compiled_rhs_cost("lenard_bernstein", 3, 6)
    # Diagonal damping stores O(n) per point, so cost should track the state
    # size (2.25x from 8 to 18 moments), not n^2.
    assert large / small < 4.0, f"diagonal operator grew {large / small:.2f}x"


# ---- from test_end_damping_physics.py ----
# Physics gate: parallel-domain end damping on a linked flux tube.
#
# Linear RHS assembly uses legacy strength ``damp_ends_amp / dt``. At the pinned
# ``dt=0.002``, interpreting 0.1 as a rate instead weakens damping 500-fold and
# allows an unphysical boundary mode to dominate (uwplasma/GKX#192).
# References below are measured compatibility regressions, not independent


# One Cyclone-like linked flux tube at the tokamak parity decks' step size. The
# ky is the top of the s-alpha deck's scan, which is where the campaign traced
# the runaway to; nperiod=2 is what gives the tube the two damped end caps at
# all, and every failing case in the parity matrix has it.
_KY_TARGET = 0.55
_NL, _NM, _NZ = 8, 24, 96
_DT = 0.002
_STEPS = 20000
_SAMPLE_STRIDE = 50
_SEED_AMPLITUDE = 1.0e-6

#: Measured with the per-step contract in place; the spurious mode gives +11.3.
_EXPECTED_GAMMA = -0.0405


def _linked_salpha_flux_tube():
    """Return the single-ky linked s-alpha tube and its geometry."""

    grid_full = build_spectral_grid(
        GridConfig(
            Nx=1,
            Ny=16,
            Nz=_NZ,
            Lx=62.8,
            Ly=62.8,
            boundary="linked",
            y0=10.0,
            ntheta=32,
            nperiod=2,
        )
    )
    ky_index = int(np.argmin(np.abs(np.asarray(grid_full.ky) - _KY_TARGET)))
    grid = select_ky_grid(grid_full, ky_index)
    geom = SAlphaGeometry.from_config(GeometryConfig(s_hat=0.8))
    return grid, geom


def test_end_damping_bounds_the_domain_end_mode_at_a_deck_step_size() -> None:
    """Bound the field/end caps and recover the pinned late-time decay rate."""

    with jax.enable_x64():
        grid, geom = _linked_salpha_flux_tube()
        params = linear_params_for_geometry(
            geom,
            tprim=2.49,
            fprim=0.8,
            damp_ends_amp=0.1,
            damp_ends_widthfrac=0.125,
            nu_hermite=1.0,
            nu_laguerre=2.0,
            hypercollisions_const=0.0,
            hypercollisions_kz=1.0,
        )
        cache = build_linear_cache(grid, geom, params, Nl=_NL, Nm=_NM)
        terms = LinearTerms(
            streaming=1.0,
            mirror=1.0,
            curvature=1.0,
            gradb=1.0,
            diamagnetic=1.0,
            collisions=0.0,
            hypercollisions=1.0,
            hyperdiffusion=0.0,
            end_damping=1.0,
            apar=0.0,
            bpar=0.0,
        )

        # Seed the middle of the tube only. The end caps are populated by
        # parallel streaming, so what the trace measures is whether the damping
        # removes that content faster than the boundary mode amplifies it.
        z = np.arange(_NZ)
        profile = np.exp(-0.5 * ((z - _NZ / 2.0) / (0.1 * _NZ)) ** 2)
        seed = np.zeros((_NL, _NM, 1, 1, _NZ), dtype=complex)
        seed[0, 0, 0, 0, :] = _SEED_AMPLITUDE * profile
        G0 = jnp.asarray(seed, dtype=jnp.complex128)

        _G, phi_t, _density = integrate_linear_diagnostics(
            G0,
            grid,
            geom,
            params,
            _DT,
            _STEPS,
            method="imex2",
            terms=terms,
            sample_stride=_SAMPLE_STRIDE,
            species_index=0,
            cache=cache,
        )

    phi = np.asarray(phi_t)[:, 0, 0, :]
    assert np.all(np.isfinite(phi)), "field history went non-finite"

    # Amplitude as a max over z rather than an L2 norm: under the rate reading
    # the field reaches ~1e163, where squaring it overflows float64 and the
    # runaway would be hidden behind an inf instead of measured.
    amplitude = np.max(np.abs(phi), axis=1)
    assert np.max(amplitude) < 1.0e-4, (
        f"peak |phi| {np.max(amplitude):.3e} ran away from the "
        f"{_SEED_AMPLITUDE:.0e} seed"
    )

    # The end caps must hold a negligible share of the mode. Under the rate
    # reading they hold order 10% of it, and then they are the mode.
    damp_profile = np.asarray(cache.damp_profile, dtype=float).reshape(-1)
    if damp_profile.size != _NZ:
        damp_profile = np.asarray(cache.linked_damp_profile, dtype=float).reshape(-1)
        damp_profile = damp_profile[:_NZ]
    end_caps = damp_profile > 0.5 * damp_profile.max()
    assert end_caps.any()
    end_share = float(np.max(np.abs(phi[-1, end_caps]))) / amplitude[-1]
    assert end_share < 1.0e-2, f"end caps hold {end_share:.3e} of the mode"

    times = _DT * _SAMPLE_STRIDE * np.arange(1, amplitude.size + 1)
    late = slice(int(0.6 * amplitude.size), None)
    gamma = float(np.polyfit(times[late], np.log(amplitude[late]), 1)[0])
    assert gamma == pytest.approx(_EXPECTED_GAMMA, abs=1.0e-2)


# ---- from test_hermite_hierarchy_physics.py ----
# Physics gates: the parallel-velocity (Hermite) hierarchy.
#
# One truncated hierarchy underlies every gate here: free streaming pushes free
# energy up in m, a hard truncation reflects it back as recurrence, and what
# the closure does with that pulse decides whether the code can report Landau
# damping, a recurrence time, or a stable design at all. The blocks below keep


def test_closure_coefficient_matches_the_analytic_form() -> None:
    """``R_{M+1}`` must follow the published expression and its limits."""

    for hermite_count in (3, 4, 6, 10, 18, 34, 64, 128):
        order = hermite_count - 1
        expected = (
            order
            / math.sqrt(2.0 * (order + 1.0))
            * math.gamma(order / 2.0)
            / math.gamma((order + 1.0) / 2.0)
        )
        assert hermite_closure_coefficient(hermite_count) == expected

        # Strictly dissipative, and approaching the asymptotic 1 - 1/(4M).
        assert 0.0 < expected < 1.0
        assert abs(expected - (1.0 - 1.0 / (4.0 * order))) < 0.05

    # The M = 2 member is exactly the Hammett-Perkins three-pole coefficient,
    # which is an independent check on the whole family.
    assert (
        abs(hermite_closure_coefficient(3) - float(np.sqrt(8.0 / np.pi) / np.sqrt(3.0)))
        < 1.0e-14
    )


def _streaming(state, hermite_count, wavenumbers, closure):
    shape = (hermite_count, 1, 1, 1)
    return np.asarray(
        linked_streaming_contribution(
            state,
            phi=jnp.zeros((1, 1, wavenumbers.size), dtype=jnp.complex128),
            apar=None,
            bpar=None,
            Jl=jnp.zeros((1, 1, 1, 1, wavenumbers.size)),
            JlB=jnp.zeros((1, 1, 1, 1, wavenumbers.size)),
            tz=jnp.asarray([1.0]),
            vth=jnp.asarray([1.0]),
            sqrt_p=jnp.sqrt(jnp.arange(1, hermite_count + 1)).reshape(shape),
            sqrt_m=jnp.sqrt(jnp.arange(0, hermite_count)).reshape(shape),
            kpar_scale=jnp.asarray(1.0),
            weight=jnp.asarray(1.0),
            kz=wavenumbers,
            dz=jnp.asarray(1.0),
            hermite_closure=closure,
        )
    )


def test_closure_acts_only_on_the_last_moment_and_dissipates() -> None:
    """The closure must be a sink confined to ``m = M``.

    Confinement is what distinguishes it from hypercollisions, which act over a
    band of high ``m``: the residual and every resolved moment are untouched,
    so the closure cannot bias the physics it is meant to protect.
    """

    hermite_count, points = 8, 32
    wavenumbers = jnp.asarray(2.0 * np.pi * np.fft.fftfreq(points, d=1.0 / points))
    generator = np.random.default_rng(0)
    state = jnp.asarray(
        generator.normal(size=(1, 1, hermite_count, 1, 1, points))
        + 1j * generator.normal(size=(1, 1, hermite_count, 1, 1, points))
    )

    truncated = _streaming(state, hermite_count, wavenumbers, "truncation")
    absorbed = _streaming(state, hermite_count, wavenumbers, "reflectionless")
    difference = absorbed - truncated

    for moment in range(hermite_count - 1):
        assert np.abs(difference[0, 0, moment]).max() == 0.0, (
            f"closure perturbed moment m={moment}, which must stay untouched"
        )
    assert np.abs(difference[0, 0, -1]).max() > 0.0

    # The added term is exactly -R sqrt(M+1) v_th |k_par| G_M.
    coefficient = hermite_closure_coefficient(hermite_count)
    expected = (
        -coefficient
        * math.sqrt(hermite_count)
        * np.asarray(abs_z_periodic(state[:, :, hermite_count - 1], kz=wavenumbers))
    )
    # The matrices are assembled in the ambient precision, so the agreement
    # floor follows it: ~1e-13 under x64, ~1e-4 under the float32 policy.
    tolerance = 1.0e-10 if jnp.zeros(1).dtype == jnp.float64 else 1.0e-3
    scale = max(float(np.abs(expected[0, 0]).max()), 1.0)
    assert np.abs(difference[0, 0, -1] - expected[0, 0]).max() < tolerance * scale

    # Strictly dissipative: it can only remove free energy from the last moment.
    production = float(np.real(np.sum(np.conj(state[0, 0, -1]) * difference[0, 0, -1])))
    assert production < 0.0


def test_truncation_remains_the_default() -> None:
    """The closure is opt-in, so existing results are unchanged."""

    hermite_count, points = 6, 16
    wavenumbers = jnp.asarray(2.0 * np.pi * np.fft.fftfreq(points, d=1.0 / points))
    generator = np.random.default_rng(1)
    state = jnp.asarray(
        generator.normal(size=(1, 1, hermite_count, 1, 1, points))
        + 1j * generator.normal(size=(1, 1, hermite_count, 1, 1, points))
    )
    shape = (hermite_count, 1, 1, 1)
    default = np.asarray(
        linked_streaming_contribution(
            state,
            phi=jnp.zeros((1, 1, points), dtype=jnp.complex128),
            apar=None,
            bpar=None,
            Jl=jnp.zeros((1, 1, 1, 1, points)),
            JlB=jnp.zeros((1, 1, 1, 1, points)),
            tz=jnp.asarray([1.0]),
            vth=jnp.asarray([1.0]),
            sqrt_p=jnp.sqrt(jnp.arange(1, hermite_count + 1)).reshape(shape),
            sqrt_m=jnp.sqrt(jnp.arange(0, hermite_count)).reshape(shape),
            kpar_scale=jnp.asarray(1.0),
            weight=jnp.asarray(1.0),
            kz=wavenumbers,
            dz=jnp.asarray(1.0),
        )
    )
    assert np.array_equal(
        default, _streaming(state, hermite_count, wavenumbers, "truncation")
    )


def test_free_streaming_conserves_norm_so_g0_cannot_exceed_one() -> None:
    """``|g_0|`` can never exceed its initial value under free streaming.

    The streaming operator is anti-Hermitian, so ``||g||`` is conserved and
    ``|g_0| <= ||g||``. This gate exists because an earlier analysis of this
    very hierarchy reported truncation "reviving" ``|g_0|`` to 8-16x the initial
    amplitude, which this bound makes impossible. Any future measurement that
    reports amplification is measuring something else -- a ratio to a
    quiescent floor, or a different normalization -- and must say so.
    """

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.artifacts.build_recurrence_closure_figure import free_streaming_revival

    for hermite in (16, 64):
        _, amplitude = free_streaming_revival(hermite, "truncation")
        assert amplitude.max() <= 1.0 + 1.0e-9, (
            f"N_m={hermite}: |g_0| reached {amplitude.max():.4f} > 1, which "
            "violates norm conservation for an anti-Hermitian operator"
        )
        # And the reflection must be essentially complete, which is the actual
        # indictment of a hard truncation: it dissipates nothing.
        assert amplitude.max() > 0.99, (
            f"N_m={hermite}: truncation revival {amplitude.max():.4f} is lower "
            "than expected for a perfectly reflecting wall"
        )


def test_absorbing_closures_beat_truncation_on_both_metrics() -> None:
    """Both absorbing treatments must suppress revival AND keep the resolved window.

    Reporting revival alone would let a closure that flattens the entire
    hierarchy look perfect, so the resolved-window error against a converged
    reference is gated alongside it.
    """

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.artifacts.build_recurrence_closure_figure import (
        measure_revival,
        resolved_window_error,
    )

    truncation_revival, _ = measure_revival(64, "truncation")
    for closure in ("hypercollisions", "reflectionless"):
        revival, _ = measure_revival(64, closure)
        error = resolved_window_error(64, closure)
        assert revival < 0.1 * truncation_revival, (
            f"{closure}: revival {revival:.4f} is not well below truncation's "
            f"{truncation_revival:.4f}"
        )
        assert error < 0.1, f"{closure}: resolved-window error {error:.3e} too large"


# ---- from test_landau_damping.py ----
# Physics gate: GKX must reproduce the exact kinetic Landau roots.
#
# The reference is the slab gyrokinetic ion-acoustic dispersion relation with
# adiabatic electrons at ``k_perp -> 0``,
#
#     1 + T_i/T_e + zeta Z(zeta) = 0,
#
# solved here from ``scipy.special.wofz`` rather than read from a table, so the
# gate cannot drift with a hard-coded constant.
#
# Three separate traps make a naive version of this test pass while measuring the
# wrong thing, and each is gated explicitly below:
#
# * a collisionless truncated Hermite system has a purely real spectrum, so it has
#   no asymptotic damping to measure at all;
# * the Landau root is not an eigenvalue of the collisional operator either -- it
#   is a pole of the continued response, reached by ``nu -> 0`` extrapolation;
# * a density perturbation with no initial flow is a standing wave, so its phase
#   does not advance and ``omega`` must come from an envelope-and-oscillation fit.


# ``test_landau_damping.py`` carried this as a module-level ``pytestmark``. It
# is applied per test here so that merging cannot extend the float64 skip to
# gates that never had it; the condition and reason are unchanged.
_LANDAU_NEEDS_FLOAT64 = pytest.mark.skipif(
    jnp.zeros(1).dtype != jnp.float64,
    reason="Landau roots need float64; CI runs JAX_ENABLE_X64",
)


@_LANDAU_NEEDS_FLOAT64
def test_landau_preview_uses_a_bounded_palette(tmp_path) -> None:
    fig, ax = plt.subplots()
    ax.plot(np.linspace(0.0, 1.0, 32), np.linspace(0.0, 1.0, 32) ** 2)
    output = tmp_path / "preview.png"

    save_figure(fig, output, palette_colors=256)

    with Image.open(output) as image:
        assert image.mode == "P"
        assert len(image.getcolors()) <= 256


@_LANDAU_NEEDS_FLOAT64
def test_collisionless_hermite_spectrum_is_purely_real() -> None:
    """Free streaming is anti-Hermitian, so a truncation cannot Landau damp.

    This is the gate that stops anyone from ever reading an asymptotic damping
    rate off a collisionless run: there is none, and whatever a fit returns is
    a transient. Getting a plausible number here would mean the streaming
    operator had acquired a spurious dissipative part.
    """

    spectrum = np.linalg.eigvals(operator_matrix(64, 0.0, 1.0))
    assert np.abs(spectrum.real).max() < 1.0e-11, (
        "collisionless Hermite spectrum is not real: "
        f"max |Re lambda| = {np.abs(spectrum.real).max():.3e}"
    )

    # With collisions it must acquire genuine damping, otherwise the gate above
    # would also pass for an operator that does nothing at all.
    collisional = np.linalg.eigvals(operator_matrix(64, 0.05, 1.0))
    assert collisional.real.min() < -1.0e-3


@_LANDAU_NEEDS_FLOAT64
@pytest.mark.parametrize(
    ("te_over_ti", "guess", "gamma_tolerance", "omega_tolerance"),
    [
        (1.0, complex(1.4, -0.6), 1.0, 0.5),
        (10.0, complex(2.6, -0.04), 1.0, 0.5),
    ],
)
def test_landau_root_recovered_by_collisional_extrapolation(
    te_over_ti: float,
    guess: complex,
    gamma_tolerance: float,
    omega_tolerance: float,
) -> None:
    """``nu -> 0`` extrapolation must land on the exact root, in percent."""

    exact = exact_root(te_over_ti, guess)
    seed = (1.0, exact.imag, exact.real, 0.0)

    # A shortened scan: the gate needs the extrapolation to be well conditioned,
    # not the figure's resolution.
    nus = _NU_SCAN[::2]
    gammas, omegas = [], []
    for nu in nus:
        times, signal = evolve(
            hermite=64, nu=float(nu), te_over_ti=te_over_ti, t_max=12.0
        )
        gamma, omega = fit_standing_wave(times, signal, (2.0, 9.0), seed)
        gammas.append(gamma)
        omegas.append(omega)

    gamma_zero = float(np.polyfit(nus, gammas, 1)[1])
    omega_zero = float(np.polyfit(nus, omegas, 1)[1])

    gamma_error = 100.0 * abs(gamma_zero - exact.imag) / abs(exact.imag)
    omega_error = 100.0 * abs(omega_zero - exact.real) / abs(exact.real)

    assert gamma_error < gamma_tolerance, (
        f"T_e/T_i={te_over_ti}: gamma {gamma_zero:.6f} vs exact {exact.imag:.6f}"
        f" ({gamma_error:.3f}%)"
    )
    assert omega_error < omega_tolerance, (
        f"T_e/T_i={te_over_ti}: omega {omega_zero:.6f} vs exact {exact.real:.6f}"
        f" ({omega_error:.3f}%)"
    )


@_LANDAU_NEEDS_FLOAT64
def test_temperature_ratio_convention_is_pinned() -> None:
    """``T_e/T_i = 10`` must not be reachable by inverting the ratio.

    GKX's ``tau_e`` is ``T_i/T_e``, the reciprocal of the ratio this test is
    written in. ``T_e = T_i`` cannot tell the two apart, which is exactly how a
    reciprocal slip survives a test suite. Pinning the asymmetric case means a
    future refactor that flips the convention fails here rather than silently
    changing every adiabatic-electron result.
    """

    exact = exact_root(10.0, complex(2.6, -0.04))
    seed = (1.0, exact.imag, exact.real, 0.0)
    times, signal = evolve(hermite=64, nu=0.005, te_over_ti=10.0, t_max=12.0)
    _, omega = fit_standing_wave(times, signal, (2.0, 9.0), seed)

    # The inverted convention puts the frequency near the T_e/T_i = 0.1 branch,
    # which is nowhere near 3.73.
    assert abs(omega - exact.real) / exact.real < 0.02, (
        f"omega {omega:.4f} is not near the T_e/T_i=10 root {exact.real:.4f};"
        " tau_e convention may have been inverted"
    )


@_LANDAU_NEEDS_FLOAT64
def test_recurrence_time_follows_the_square_root_law() -> None:
    """The revival must appear at ``t_rec ~ 2 sqrt(N_m)``, not at ``~N_m``.

    This distinguishes a genuine Hermite recurrence from a generic instability:
    the scaling in ``N_m`` is the fingerprint, and it is what makes adding
    moments a weak remedy.
    """

    observed = []
    for hermite in (16, 64):
        times, signal = evolve(
            hermite=hermite, nu=0.0, te_over_ti=1.0, t_max=3.0 * np.sqrt(hermite)
        )
        envelope = np.abs(signal)
        predicted = 2.0 * np.sqrt(hermite)
        # Look for the revival in a window around the predicted time, compared
        # against the quiet stretch that precedes it.
        search = (times > 0.55 * predicted) & (times < 1.6 * predicted)
        quiet = (times > 0.3 * predicted) & (times <= 0.55 * predicted)
        assert envelope[search].max() > 3.0 * envelope[quiet].mean(), (
            f"N_m={hermite}: no revival found near t_rec={predicted:.1f}"
        )
        observed.append(times[search][np.argmax(envelope[search])] / predicted)

    # Both resolutions must locate the revival at the same multiple of
    # 2 sqrt(N_m); a linear-in-N_m law would make these differ by a factor 2.
    assert abs(observed[0] - observed[1]) < 0.35, (
        f"revival does not scale as sqrt(N_m): ratios {observed}"
    )


# ---- from test_objective_reports_stability.py ----
# Physics: can the linear objective report that a design is stable?
#
# An optimizer follows the objective, so an objective with a floor at zero is not
# merely imprecise -- it can never say "this design is quiet", and wherever the
# physical branch is weak it returns whatever marginal mode happens to sit highest.
#
# That is what the shipped defaults did. Every dissipation amplitude in
# ``_default_gradient_linear_params`` was zero, which leaves a truncated Hermite
# hierarchy with no velocity-space dissipation, so at zero drive the whole spectrum
# sits on the imaginary axis and ``argmax(Re lambda)`` is floored at zero or above.
# Measured on Cyclone at zero drive: ``+9.3e-05`` at ``N_m = 8``, ``+6.4e-14`` at
# 16, ``+1.0e-13`` at 32, at ``|omega|`` between 20 and 90.
#
# Zero drive is the sharpest available check because it needs no reference value.
# With ``tprim = fprim = 0`` there is no free energy in the system, so every mode
# must be damped, and any positive growth rate is numerical by construction.
#
# These would fail if the closure were removed from the defaults, and the
# convergence case would fail if it were replaced by hyperdiffusion, which in this
# path is exactly ``-D_hyper * I`` and so shifts every eigenvalue by a constant
# without restoring resolution.


HERMITE_LADDER = (8, 16, 32)


@pytest.fixture(scope="module")
def cyclone_geometry():
    analytic = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778)
    theta = jnp.linspace(-jnp.pi, jnp.pi, 24, endpoint=False)
    return sample_flux_tube_geometry(analytic, theta)


def _growth(geometry, *, n_hermite: int, drive: float) -> tuple[float, float]:
    params = dataclasses.replace(
        _default_gradient_linear_params(geometry),
        tprim=2.49 * drive,
        fprim=0.8 * drive,
    )
    values = solver_objective_vector_from_geometry(
        geometry,
        selected_ky_index=1,
        n_laguerre=2,
        n_hermite=n_hermite,
        ny=12,
        ly=62.83,
        params_linear=params,
    )
    return float(values[0]), float(values[1])


@pytest.mark.parametrize("n_hermite", HERMITE_LADDER)
def test_zero_drive_is_damped_at_every_hermite_truncation(cyclone_geometry, n_hermite):
    """No free energy, so no growth -- at every truncation, not just coarse ones."""

    growth, _ = _growth(cyclone_geometry, n_hermite=n_hermite, drive=0.0)

    assert growth < 0.0, (
        f"objective reports growth {growth:+.4e} at zero drive with "
        f"n_hermite={n_hermite}; with no gradient there is no free energy, so a "
        "non-negative value means the branch selector is returning an "
        "undamped numerical mode"
    )


def test_the_unstable_branch_converges_in_hermite(cyclone_geometry):
    """Refining velocity space must improve the answer, not replace it."""

    measured = [
        _growth(cyclone_geometry, n_hermite=n, drive=1.0) for n in HERMITE_LADDER
    ]
    growths = [item[0] for item in measured]
    frequencies = [item[1] for item in measured]

    assert all(value > 0.0 for value in growths), (
        f"Cyclone at nominal drive must be unstable, got {growths}"
    )
    # One branch, not three: the frequency identifies the mode, and a selector
    # that wandered between branches would move it far more than this.
    assert max(frequencies) - min(frequencies) < 0.05, (
        f"frequency moved across the Hermite ladder ({frequencies}), so the "
        "resolutions are not describing the same mode"
    )
    spread = (max(growths) - min(growths)) / max(growths)
    assert spread < 0.10, (
        f"growth rates {growths} span {spread:.1%} across n_hermite="
        f"{HERMITE_LADDER}; the truncation is not converged"
    )


# ---- from test_validation_gates.py ----
# Physics gates: the validation-gate machinery and the zonal-response metrics it gates.
#
# These are the measurement and pass/fail primitives that every tracked
# comparison runs through -- scalar and family gate reports with their inclusive
# thresholds and fail-closed guards, and the W7-X/Merlo zonal-response trace
# loaders, reference tables, and GAM residual/damping/frequency estimators whose


def test_validation_gate_facade_points_to_focused_modules() -> None:
    import gkx.diagnostics.analysis as metric_analysis
    import gkx.diagnostics.modes as mode_analysis
    import scripts.checks._gates.validation_gates as gates

    assert LateTimeLinearMetrics is metric_analysis.LateTimeLinearMetrics
    assert NonlinearWindowMetrics is metric_analysis.NonlinearWindowMetrics
    assert (
        EigenfunctionComparisonMetrics is mode_analysis.EigenfunctionComparisonMetrics
    )
    assert gates.evaluate_scalar_gate is evaluate_scalar_gate
    assert gates.gate_report_to_dict is gate_report_to_dict
    assert gates.observed_order_gate_report is observed_order_gate_report


def test_validation_gate_primitives_are_public_and_owned_by_diagnostics() -> None:
    metrics = zonal_flow_response_metrics(
        np.linspace(0.0, 2.0, 8), np.linspace(1.0, 0.6, 8)
    )
    assert isinstance(metrics, ZonalFlowResponseMetrics)
    assert validation_gates.observed_order_gate_report is observed_order_gate_report
    assert (
        validation_gates.branch_continuity_gate_report is branch_continuity_gate_report
    )
    assert (
        validation_gates.nonlinear_heat_flux_convergence_gate_report
        is nonlinear_heat_flux_convergence_gate_report
    )


def test_scalar_gate_and_json_report_are_strict_and_serializable() -> None:
    passed = evaluate_scalar_gate("gamma", 1.01, 1.0, atol=0.0, rtol=0.02)
    failed = evaluate_scalar_gate("omega", 0.7, 1.0, atol=0.0, rtol=0.02)
    near_zero = evaluate_scalar_gate(
        "zonal_residual", 1.0e-4, 0.0, atol=2.0e-4, rtol=0.0
    )
    report = gate_report("case", "reference", [passed, failed, near_zero])
    payload = gate_report_to_dict(report)

    assert isinstance(passed, ScalarGateResult)
    assert isinstance(report, GateReport)
    assert report.passed is False
    assert payload["gates"][0]["metric"] == "gamma"
    assert payload["gates"][1]["passed"] is False
    assert payload["gates"][2]["rel_error"] is None
    json.dumps(payload, allow_nan=False)

    with pytest.raises(ValueError):
        gate_report("empty", "reference", [])
    with pytest.raises(ValueError):
        evaluate_scalar_gate("bad", 1.0, 1.0, atol=-1.0, rtol=0.0)


def test_scalar_gate_thresholds_are_inclusive_and_nonfinite_values_fail() -> None:
    exact_combined = evaluate_scalar_gate(
        "combined_tol", 1.25, 1.0, atol=0.05, rtol=0.20
    )
    just_over = evaluate_scalar_gate("combined_tol", 1.2501, 1.0, atol=0.05, rtol=0.20)
    exact_zero_ref = evaluate_scalar_gate(
        "zero_ref", -2.0e-4, 0.0, atol=2.0e-4, rtol=0.0
    )
    just_over_zero_ref = evaluate_scalar_gate(
        "zero_ref", 2.01e-4, 0.0, atol=2.0e-4, rtol=0.0
    )

    assert exact_combined.passed is True
    assert just_over.passed is False
    assert exact_zero_ref.passed is True
    assert just_over_zero_ref.passed is False

    nonfinite_report = gate_report(
        "nonfinite",
        "synthetic",
        (
            evaluate_scalar_gate("nan_observed", np.nan, 1.0, atol=1.0, rtol=0.0),
            evaluate_scalar_gate("inf_observed", np.inf, 1.0, atol=1.0, rtol=0.0),
            evaluate_scalar_gate("inf_reference", 1.0, np.inf, atol=1.0, rtol=0.0),
        ),
    )
    payload = gate_report_to_dict(nonfinite_report)

    assert nonfinite_report.passed is False
    assert np.isinf(nonfinite_report.max_abs_error)
    assert all(gate.passed is False for gate in nonfinite_report.gates)
    assert payload["max_abs_error"] is None
    assert payload["gates"][0]["observed"] is None
    assert payload["gates"][1]["observed"] is None
    assert payload["gates"][2]["reference"] is None
    json.dumps(payload, allow_nan=False)


def test_family_gate_thresholds_are_inclusive_at_documented_bounds() -> None:
    eigen = eigenfunction_gate_report(
        EigenfunctionComparisonMetrics(overlap=0.95, relative_l2=0.25, phase_shift=0.0),
        case="mode",
        source="synthetic",
        min_overlap=0.95,
        max_relative_l2=0.25,
    )
    order = observed_order_gate_report(
        ObservedOrderMetrics(
            step_sizes=np.array([0.4, 0.2, 0.1]),
            errors=np.array([4.0e-3, 2.0e-3, 1.0e-3]),
            orders=np.array([1.5, 1.5]),
            asymptotic_order=2.0,
        ),
        case="order",
        source="synthetic",
        min_asymptotic_order=2.0,
        min_pairwise_order=1.5,
        max_final_error=1.0e-3,
    )
    branch = branch_continuity_gate_report(
        BranchContinuationMetrics(
            ky=np.array([0.1, 0.2, 0.3]),
            gamma=np.array([0.1, 0.15, 0.2]),
            omega=np.array([1.0, 1.1, 1.2]),
            rel_gamma_jumps=np.array([0.5, 0.25]),
            rel_omega_jumps=np.array([0.25, 0.1]),
            max_rel_gamma_jump=0.5,
            max_rel_omega_jump=0.25,
            min_successive_overlap=0.95,
        ),
        case="branch",
        source="synthetic",
        max_rel_gamma_jump=0.5,
        max_rel_omega_jump=0.25,
        min_successive_overlap=0.95,
    )

    assert eigen.passed is True
    assert order.passed is True
    assert branch.passed is True


def test_validation_gate_family_helpers_cover_physics_observables() -> None:
    linear = LateTimeLinearMetrics(
        gamma_fit=1.0,
        omega_fit=2.0,
        gamma_tail_mean=1.0,
        omega_tail_mean=2.0,
        gamma_tail_std=0.01,
        omega_tail_std=0.02,
        tmin=1.0,
        tmax=2.0,
        nsamples=10,
        signal_source="mode",
    )
    nonlinear = NonlinearWindowMetrics(
        tmin=1.0,
        tmax=2.0,
        nsamples=10,
        heat_flux_mean=1.0,
        heat_flux_std=0.1,
        heat_flux_rms=1.05,
        wphi_mean=2.0,
        wphi_std=0.2,
        wg_mean=3.0,
        wg_std=0.3,
        phi_mode_envelope_mean=4.0,
        phi_mode_envelope_std=0.4,
        phi_mode_envelope_max=4.5,
    )
    nonlinear_convergence = NonlinearHeatFluxConvergenceMetrics(
        tmin=10.0,
        tmax=20.0,
        nsamples=12,
        # Explicit: the convergence gate divides the standard error by the
        # INDEPENDENT sample count, and a metrics object built by hand defaults
        # it to zero so it cannot pass a statistical gate by omission.
        n_eff=12.0,
        tau_ac=0.0,
        heat_flux_mean=1.0,
        heat_flux_std=0.02,
        heat_flux_cv=0.02,
        heat_flux_rms=1.0002,
        terminal_tmin=15.0,
        terminal_tmax=20.0,
        terminal_nsamples=6,
        terminal_heat_flux_mean=1.01,
        mean_rel_delta=0.01,
        trend=0.02,
        abs_trend=0.02,
        start_fraction=0.5,
        terminal_fraction=0.5,
    )
    zonal = ZonalFlowResponseMetrics(
        initial_level=1.0,
        initial_policy="first_abs",
        residual_level=0.2,
        residual_std=0.01,
        response_rms=0.3,
        gam_frequency=2.0,
        gam_damping_rate=0.1,
        damping_method="branchwise_extrema",
        frequency_method="hilbert_phase",
        peak_count=4,
        peak_fit_count=4,
        tmin=0.0,
        tmax=10.0,
        fit_tmin=0.0,
        fit_tmax=5.0,
        peak_times=np.array([1.0, 2.0]),
        peak_envelope=np.array([0.5, 0.4]),
        max_peak_times=np.array([1.0]),
        max_peak_values=np.array([0.5]),
        min_peak_times=np.array([2.0]),
        min_peak_values=np.array([-0.4]),
    )

    assert (
        linear_metrics_gate_report(linear, linear, case="linear", source="self").passed
        is True
    )
    assert (
        nonlinear_window_gate_report(
            nonlinear, nonlinear, case="nonlinear", source="self"
        ).passed
        is True
    )
    assert (
        nonlinear_heat_flux_convergence_gate_report(
            nonlinear_convergence,
            case="nonlinear_convergence",
            source="self",
            max_mean_rel_delta=0.02,
            max_cv=0.03,
            max_abs_trend=0.03,
            min_samples=12,
        ).passed
        is True
    )
    assert (
        zonal_response_gate_report(
            zonal,
            zonal,
            case="zonal",
            source="self",
            residual_atol=0.0,
            frequency_atol=0.0,
            damping_atol=0.0,
        ).passed
        is True
    )
    assert (
        eigenfunction_gate_report(
            EigenfunctionComparisonMetrics(
                overlap=0.99, relative_l2=0.01, phase_shift=0.0
            ),
            case="mode",
            source="self",
        ).passed
        is True
    )


def test_order_and_branch_gates_preserve_open_lane_failures() -> None:
    observed = ObservedOrderMetrics(
        step_sizes=np.array([0.4, 0.2, 0.1]),
        errors=np.array([0.01, 0.02, 0.002]),
        orders=np.array([-1.0, 3.32192809]),
        asymptotic_order=3.32192809,
    )
    order_report = observed_order_gate_report(
        observed,
        case="nonmonotone",
        source="synthetic",
        min_asymptotic_order=1.0,
        min_pairwise_order=0.0,
    )
    assert order_report.passed is False
    assert order_report.gates[1].metric == "min_pairwise_order_deficit"

    branch = BranchContinuationMetrics(
        ky=np.array([0.1, 0.2]),
        gamma=np.array([0.1, 0.3]),
        omega=np.array([1.0, 1.1]),
        rel_gamma_jumps=np.array([0.666]),
        rel_omega_jumps=np.array([0.091]),
        max_rel_gamma_jump=0.666,
        max_rel_omega_jump=0.091,
        min_successive_overlap=None,
    )
    branch_report = branch_continuity_gate_report(
        branch,
        case="branch",
        source="synthetic",
        max_rel_gamma_jump=0.5,
        max_rel_omega_jump=0.5,
        min_successive_overlap=0.95,
    )
    assert branch_report.passed is False
    assert branch_report.gates[-1].metric == "successive_overlap_deficit"


def test_nonlinear_window_gate_optional_envelope_policy_is_explicit() -> None:
    reference = NonlinearWindowMetrics(
        tmin=1.0,
        tmax=2.0,
        nsamples=8,
        heat_flux_mean=1.0,
        heat_flux_std=0.1,
        heat_flux_rms=1.1,
        wphi_mean=2.0,
        wphi_std=0.2,
        wg_mean=3.0,
        wg_std=0.3,
        phi_mode_envelope_mean=1.0,
        phi_mode_envelope_std=0.1,
        phi_mode_envelope_max=1.2,
    )
    envelope_mismatch = NonlinearWindowMetrics(
        tmin=1.0,
        tmax=2.0,
        nsamples=8,
        heat_flux_mean=1.0,
        heat_flux_std=0.1,
        heat_flux_rms=1.1,
        wphi_mean=2.0,
        wphi_std=0.2,
        wg_mean=3.0,
        wg_std=0.3,
        phi_mode_envelope_mean=2.0,
        phi_mode_envelope_std=0.1,
        phi_mode_envelope_max=2.2,
    )
    unresolved_mode = NonlinearWindowMetrics(
        tmin=1.0,
        tmax=2.0,
        nsamples=8,
        heat_flux_mean=1.0,
        heat_flux_std=0.1,
        heat_flux_rms=1.1,
        wphi_mean=2.0,
        wphi_std=0.2,
        wg_mean=3.0,
        wg_std=0.3,
        phi_mode_envelope_mean=None,
        phi_mode_envelope_std=None,
        phi_mode_envelope_max=None,
    )

    envelope_report = nonlinear_window_gate_report(
        envelope_mismatch,
        reference,
        case="window",
        source="synthetic",
        rtol=0.1,
    )
    excluded_report = nonlinear_window_gate_report(
        envelope_mismatch,
        reference,
        case="window",
        source="synthetic",
        rtol=0.1,
        include_envelope=False,
    )
    unresolved_report = nonlinear_window_gate_report(
        unresolved_mode,
        reference,
        case="window",
        source="synthetic",
        rtol=0.1,
    )

    assert envelope_report.passed is False
    assert envelope_report.gates[-1].metric == "phi_mode_envelope_mean"
    assert excluded_report.passed is True
    assert [gate.metric for gate in excluded_report.gates] == [
        "heat_flux_mean",
        "heat_flux_rms",
        "wphi_mean",
        "wg_mean",
    ]
    assert unresolved_report.passed is True
    assert len(unresolved_report.gates) == 4


def test_validation_gate_threshold_guards_are_fail_closed() -> None:
    convergence = NonlinearHeatFluxConvergenceMetrics(
        tmin=10.0,
        tmax=20.0,
        nsamples=8,
        heat_flux_mean=1.0,
        heat_flux_std=0.1,
        heat_flux_cv=0.1,
        heat_flux_rms=1.01,
        terminal_tmin=15.0,
        terminal_tmax=20.0,
        terminal_nsamples=4,
        terminal_heat_flux_mean=1.02,
        mean_rel_delta=0.02,
        trend=0.03,
        abs_trend=0.03,
        start_fraction=0.5,
        terminal_fraction=0.5,
    )
    order = ObservedOrderMetrics(
        step_sizes=np.array([0.4, 0.2]),
        errors=np.array([0.02, 0.01]),
        orders=np.array([1.0]),
        asymptotic_order=1.0,
    )
    branch = BranchContinuationMetrics(
        ky=np.array([0.1, 0.2]),
        gamma=np.array([0.1, 0.2]),
        omega=np.array([1.0, 1.1]),
        rel_gamma_jumps=np.array([0.5]),
        rel_omega_jumps=np.array([0.1]),
        max_rel_gamma_jump=0.5,
        max_rel_omega_jump=0.1,
        min_successive_overlap=0.9,
    )

    with pytest.raises(ValueError, match="non-negative"):
        nonlinear_heat_flux_convergence_gate_report(
            convergence,
            case="heat",
            source="synthetic",
            max_cv=-0.1,
        )
    with pytest.raises(ValueError, match="min_samples"):
        nonlinear_heat_flux_convergence_gate_report(
            convergence,
            case="heat",
            source="synthetic",
            min_samples=0,
        )
    with pytest.raises(ValueError, match="min_overlap"):
        eigenfunction_gate_report(
            EigenfunctionComparisonMetrics(
                overlap=0.9,
                relative_l2=0.1,
                phase_shift=0.0,
            ),
            case="mode",
            source="synthetic",
            min_overlap=1.01,
        )
    with pytest.raises(ValueError, match="relative_l2"):
        eigenfunction_gate_report(
            EigenfunctionComparisonMetrics(
                overlap=0.9,
                relative_l2=0.1,
                phase_shift=0.0,
            ),
            case="mode",
            source="synthetic",
            max_relative_l2=-0.1,
        )
    with pytest.raises(ValueError, match="min_asymptotic_order"):
        observed_order_gate_report(
            order,
            case="order",
            source="synthetic",
            min_asymptotic_order=-1.0,
        )
    with pytest.raises(ValueError, match="min_pairwise_order"):
        observed_order_gate_report(
            order,
            case="order",
            source="synthetic",
            min_asymptotic_order=1.0,
            min_pairwise_order=-1.0,
        )
    with pytest.raises(ValueError, match="max_final_error"):
        observed_order_gate_report(
            order,
            case="order",
            source="synthetic",
            min_asymptotic_order=1.0,
            max_final_error=-1.0,
        )
    with pytest.raises(ValueError, match="maximum relative jumps"):
        branch_continuity_gate_report(
            branch,
            case="branch",
            source="synthetic",
            max_rel_gamma_jump=-0.1,
            max_rel_omega_jump=0.2,
        )
    with pytest.raises(ValueError, match="min_successive_overlap"):
        branch_continuity_gate_report(
            branch,
            case="branch",
            source="synthetic",
            max_rel_gamma_jump=1.0,
            max_rel_omega_jump=1.0,
            min_successive_overlap=-0.1,
        )


# ---- from test_zonal_validation.py ----


GAMMA_GATE_ATOL_R0_OVER_VI = 0.03


MERLO_R0 = 2.77778


def _merlo_like_gam_trace(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """A damped GAM on a residual, plus the ripple that broke the old estimator.

    The second term is a weakly damped, higher-frequency ripple -- what
    velocity-space recurrence actually adds to a collisionless Hermite trace. It
    puts shallow extra extrema inside the fit window, and a four-extrema
    log-linear fit resolves them at one output cadence and not at another. That
    is the mechanism that moved the shipped gamma_GAM by 52 percent when only
    the diagnostic sample spacing changed, at identical physics.
    """

    t = np.linspace(0.0, 60.0, n_samples)
    gam = 0.2 + np.exp(-0.066 * t) * np.cos(0.845 * t)
    ripple = 0.08 * np.exp(-0.02 * t) * np.cos(4.2 * t + 2.1)
    return t, gam + ripple


def _gamma_r0_over_vi(t: np.ndarray, y: np.ndarray, mode: str) -> float:
    metrics = zonal_flow_response_metrics(
        t,
        y,
        tail_fraction=0.3,
        initial_policy="first_abs",
        peak_fit_max_peaks=4,
        damping_fit_mode=mode,
        frequency_fit_mode="hilbert_phase",
        fit_window_tmax=30.0,
    )
    return -float(metrics.gam_damping_rate) * MERLO_R0


def test_period_rms_damping_is_independent_of_the_diagnostic_output_cadence() -> None:
    """The gated GAM damping must not move when only the output cadence changes.

    Output cadence carries no physics: the same trajectory written out twice as
    often has to give the same damping rate. The retired branchwise-extrema fit
    did not -- it moved by more than its own gate tolerance, so its PASS was a
    property of ``sample_stride`` rather than of the solver. This pins that the
    period-RMS envelope estimator is flat across a 4x cadence span while the
    extrema fit is not.
    """

    cadences = [(4801, "0.0125"), (2401, "0.025"), (1201, "0.05")]
    envelope = []
    extrema = []
    for n_samples, _label in cadences:
        t, y = _merlo_like_gam_trace(n_samples)
        envelope.append(_gamma_r0_over_vi(t, y, "period_rms_envelope"))
        extrema.append(_gamma_r0_over_vi(t, y, "branchwise_extrema"))

    envelope_spread = max(envelope) - min(envelope)
    extrema_spread = max(extrema) - min(extrema)

    assert envelope_spread < 0.1 * GAMMA_GATE_ATOL_R0_OVER_VI
    assert extrema_spread > GAMMA_GATE_ATOL_R0_OVER_VI
    assert all(np.isfinite(envelope))


def test_period_rms_damping_recovers_a_known_decay_rate() -> None:
    """The estimator has to be right, not only stable."""

    t = np.linspace(0.0, 60.0, 4801)
    for truth in (0.03, 0.06, 0.10):
        response = 0.2 + np.exp(-truth * t) * np.cos(0.8 * t)
        metrics = zonal_flow_response_metrics(
            t,
            response,
            initial_policy="first_abs",
            damping_fit_mode="period_rms_envelope",
            frequency_fit_mode="hilbert_phase",
            fit_window_tmax=30.0,
        )
        # A sliding one-period RMS reads a decaying sinusoid about 1-2 percent
        # low, because one window of a decaying signal is not one window of a
        # stationary one. That bias is deterministic and far inside the gate.
        assert metrics.gam_damping_rate == pytest.approx(truth, rel=0.03)
        assert metrics.damping_fit_tmax > metrics.damping_fit_tmin


def test_period_rms_damping_ignores_a_slowly_drifting_offset() -> None:
    """A drifting oscillation centre must not leak into the damping rate.

    The residual subtraction uses one number for the whole trace, so at early
    times the oscillation is not centred on it. Branch-wise extrema fits inherit
    that offset; a sliding one-period mean removes it wherever it sits.
    """

    t = np.linspace(0.0, 60.0, 4801)
    oscillation = np.exp(-0.06 * t) * np.cos(0.8 * t)
    flat = zonal_flow_response_metrics(
        t,
        0.2 + oscillation,
        initial_policy="first_abs",
        damping_fit_mode="period_rms_envelope",
        frequency_fit_mode="hilbert_phase",
        fit_window_tmax=30.0,
    )
    drifting = zonal_flow_response_metrics(
        t,
        0.2 + 0.08 * np.exp(-0.05 * t) + oscillation,
        initial_policy="first_abs",
        damping_fit_mode="period_rms_envelope",
        frequency_fit_mode="hilbert_phase",
        fit_window_tmax=30.0,
    )

    shift = abs(drifting.gam_damping_rate - flat.gam_damping_rate) * MERLO_R0
    assert shift < 0.1 * GAMMA_GATE_ATOL_R0_OVER_VI


# ---- the no-argument demo reports its own eigenvalue ----------------------
#
# The demo is the first thing most people run, so the number it prints is the
# first evidence anyone has about whether this code works. It is a coarse
# configuration on purpose (Nl = 7, Nm = 14), which is fine: coarse has a right
# answer too, and that answer is the eigenvalue of the operator the demo builds.
# What is not fine is printing a growth rate that disagrees with it. At the
# original dt = 0.03 over 500 steps the demo printed 0.089982 against a
# certified 0.103263, 12.9 percent low, and said so in four warnings that a
# newcomer has no way to weigh.
#
# This gate runs the demo's own settings twice -- time integration, as the demo
# does, and the certified Krylov eigensolver -- and requires them to agree. It
# is the check that would have caught the drift, and it fails if anyone shortens
# the horizon or raises the step back over the CFL bound.


def _demo_case(tmp_path):
    """Write and load the demo's deck exactly as the demo itself does."""

    from gkx.cli import default_demo_toml_text

    deck = tmp_path / "demo.toml"
    deck.write_text(default_demo_toml_text(), encoding="utf-8")
    return deck


def test_the_demo_reports_the_eigenvalue_of_the_case_it_builds(tmp_path) -> None:
    from gkx.cli import DEFAULT_DEMO_SETTINGS
    from gkx.runtime import run_runtime_linear
    from gkx.workflows.runtime.toml import load_runtime_from_toml

    settings = DEFAULT_DEMO_SETTINGS
    cfg, _data = load_runtime_from_toml(_demo_case(tmp_path))

    shared = dict(
        ky_target=float(settings["ky"]),
        Nl=int(settings["Nl"]),
        Nm=int(settings["Nm"]),
    )
    integrated = run_runtime_linear(
        cfg,
        solver="time",
        method=str(settings["method"]),
        dt=float(settings["dt"]),
        steps=int(settings["steps"]),
        sample_stride=int(settings["sample_stride"]),
        **shared,
    )
    certified = run_runtime_linear(cfg, solver="krylov", **shared)

    assert certified.gamma > 0.0, "the demo case should be unstable"
    relative = abs(integrated.gamma - certified.gamma) / abs(certified.gamma)
    assert relative < 0.01, (
        f"the demo's time-integrated gamma {integrated.gamma:.6f} disagrees with "
        f"the certified eigenvalue {certified.gamma:.6f} of the same case by "
        f"{100 * relative:.1f}%. The demo is the first number anyone sees from "
        "this code; if its horizon is too short or its step is over the CFL "
        "bound, it prints a biased growth rate. Restore dt and steps rather "
        "than widening this tolerance."
    )
