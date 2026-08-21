"""Physics gate: the multispecies finite-Larmor Coulomb operator reduces to the
drift-kinetic multispecies Coulomb operator as the Bessel argument ``b -> 0``.

This is the foundational reduction limit for the multispecies gyrokinetic
Coulomb collision operator. The finite-wavelength pair generation
(``build_finite_wavelength_coulomb_pair_tables``) accepts an arbitrary
mass ratio ``sigma`` and temperature ratio ``tau`` (Frei, Ball, Hoffmann,
Jorge, Ricci & Stenger 2021, arXiv:2104.11480, Eqs. 3.47-3.50). As
``b_a = k_perp v_{th,a}/Omega_a -> 0`` it must recover the drift-kinetic
multispecies Coulomb operator of Jorge et al. (2018), which GKX generates via
``coulomb_drift_kinetic_moment_matrices`` (arXiv:2104.11480, Eqs. 3.55-3.56).

The finite-wavelength tables use the signed-Laguerre runtime convention, so the
comparison applies the ``(-1)^lag (x) (-1)^lag`` sign transform to the
finite-wavelength blocks before matching. Verified for like-species,
electron-ion, and arbitrary unequal pairs -- closing the "unequal-species
finite-wavelength Coulomb is unvalidated" gap.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "artifacts"))

from build_linear_validation_artifacts import (  # noqa: E402
    build_finite_wavelength_coulomb_pair_tables,
    coulomb_drift_kinetic_moment_matrices,
)


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
