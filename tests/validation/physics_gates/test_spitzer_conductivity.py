"""Physics gate: the Coulomb operator reproduces the Spitzer-Harm conductivity.

The stationary Spitzer problem drives a uniform parallel electric field against
electron collisions and asks for the resulting current. It is the cheapest
end-to-end test of a collision operator that has a closed-form answer, and it
exercises the unlike-species (electron-ion) coefficients that the like-species
conservation gates cannot reach.

Electrons collide with themselves and with a fixed Maxwellian ion background of
charge ``Z``. Writing the steady moment balance as ``C N + s = 0``, with ``s``
the linearized parallel-field drive that survives only in ``(p, j) = (1, 0)``,
the parallel flow follows from ``u_e = N^{10} v_Te / sqrt(2)``.

Quasineutrality ``n_i Z = n_e`` makes the electron-ion collision frequency scale
as ``nu_ei ~ n_i Z^2 = n_e Z``, so the total operator is
``C_ee^T + C_ee^F + Z C_ei^T`` in units of ``nu_ee``.

The absolute conductivity depends on the normalization convention, of which
three incompatible ones appear in this literature, so this gate asserts the
*ratio*

    gamma_E(Z) = sigma(e-e and e-i) / sigma(e-i only)

which is convention free. It is the classic Spitzer-Harm correction to the
Lorentz-gas conductivity, tabulated in Spitzer & Harm, *Phys. Rev.* 89, 977
(1953), and it must approach unity as ``Z -> infinity`` because the
electron-ion term then dominates.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "artifacts"))

from build_linear_validation_artifacts import (  # noqa: E402
    coulomb_drift_kinetic_moment_matrices,
)

from gkx.operators.linear.collisions import (  # noqa: E402
    solve_driven_collision_response,
)

# Spitzer & Harm (1953), Table III: the electron-electron correction to the
# Lorentz conductivity.
SPITZER_HARM_GAMMA_E = {1: 0.5816, 2: 0.6833, 4: 0.7849, 16: 0.9225}

# (P, J) = (7, 2) is converged to better than 0.3% against (9, 3) while keeping
# the multiprecision generation near a second.
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
