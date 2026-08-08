#!/usr/bin/env python3
"""Generate linear-validation figures and gate reports.

Subcommands:
  collision-table Generate checked high-precision collision coefficient data.
  collision-verification
                  Build the Coulomb algebra/convergence verification panel.
  collision-response
                  Build the drift-kinetic driven-response convergence panel.
  collision-itg   Build the finite-wavelength slab-ITG convergence panel.
  collision-endpoint
                  Generate one equal-species finite-wavelength Coulomb archive.
  collision-project-table
                  Project a complete table onto a smaller moment basis.
  figures         Build Cyclone, ETG, and KBM comparison figures.
  observed-order  Build a convergence observed-order JSON/plot gate.
  kbm-branch      Build a KBM branch-continuity JSON gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from gkx.benchmarking.shared import LinearScanResult

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OBSERVED_CSV = REPO_ROOT / "docs" / "_static" / "cyclone_resolution_subset.csv"
DEFAULT_OBSERVED_JSON = (
    REPO_ROOT / "docs" / "_static" / "cyclone_resolution_observed_order.json"
)
DEFAULT_OBSERVED_PNG = (
    REPO_ROOT / "docs" / "_static" / "cyclone_resolution_observed_order.png"
)
DEFAULT_KBM_CANDIDATES = (
    REPO_ROOT / "docs" / "_static" / "comparison" / "kbm_reference_candidates.csv"
)
DEFAULT_KBM_BRANCH_OUT = REPO_ROOT / "docs" / "_static" / "kbm_branch_gate_summary.json"
DEFAULT_COLLISION_TABLE = (
    REPO_ROOT / "src" / "gkx" / "data" / "advanced_collision_six_moment.npy"
)
DEFAULT_COLLISION_METADATA = DEFAULT_COLLISION_TABLE.with_suffix(".json")
DEFAULT_COLLISION_VERIFICATION_JSON = (
    REPO_ROOT / "docs" / "_static" / "collision_operator_verification.json"
)
DEFAULT_COLLISION_VERIFICATION_PNG = (
    REPO_ROOT / "docs" / "_static" / "collision_operator_verification.png"
)
DEFAULT_COLLISION_RESPONSE_JSON = (
    REPO_ROOT / "docs" / "_static" / "collision_response_convergence.json"
)
DEFAULT_COLLISION_RESPONSE_CSV = DEFAULT_COLLISION_RESPONSE_JSON.with_suffix(".csv")
DEFAULT_COLLISION_RESPONSE_PNG = DEFAULT_COLLISION_RESPONSE_JSON.with_suffix(".png")
DEFAULT_COLLISION_ITG_JSON = (
    REPO_ROOT / "docs" / "_static" / "collision_finite_wavelength_itg_convergence.json"
)
DEFAULT_COLLISION_ITG_PNG = DEFAULT_COLLISION_ITG_JSON.with_suffix(".png")


def _coulomb_e_mp(order: int, chi: Any, mp: Any) -> Any:
    half = mp.mpf("0.5")
    return mp.gamma(order + half) / (mp.gamma(half) * (1 + chi * chi) ** (order + half))


def _coulomb_E_mp(order: int, chi: Any, mp: Any) -> Any:
    return (
        chi
        / 2
        * sum(
            mp.factorial(order) / mp.factorial(index) * _coulomb_e_mp(index, chi, mp)
            for index in range(order + 1)
        )
    )


def _cached_coulomb_integrals_mp(
    chi: Any, mp: Any
) -> tuple[Callable[[int], Any], Callable[[int], Any]]:
    """Return call-local Appendix-A integrals with the exact E-order recurrence."""

    @cache
    def coulomb_e(order: int) -> Any:
        return _coulomb_e_mp(order, chi, mp)

    @cache
    def coulomb_E(order: int) -> Any:
        if order == 0:
            return chi * coulomb_e(0) / 2
        return order * coulomb_E(order - 1) + chi * coulomb_e(order) / 2

    return coulomb_e, coulomb_E


def coulomb_speed_integrals(
    max_order: int,
    chi: float,
    *,
    digits: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return the Appendix-A Coulomb speed integrals ``e_k`` and ``E_k``.

    ``chi`` is the thermal-speed ratio.  The recurrence-free multiprecision
    formulas implement Frei et al. (2021), equations (A8a)--(A8b), and are
    intended for offline coefficient generation rather than runtime kernels.
    """

    if max_order < 0:
        raise ValueError("max_order must be >= 0")
    if chi <= 0.0 or not math.isfinite(chi):
        raise ValueError("chi must be finite and > 0")
    if digits < 16:
        raise ValueError("digits must be >= 16")

    import mpmath as mp

    with mp.workdps(digits):
        chi_mp = mp.mpf(chi)
        e_values = [_coulomb_e_mp(order, chi_mp, mp) for order in range(max_order + 1)]
        E_values = [_coulomb_E_mp(order, chi_mp, mp) for order in range(max_order + 1)]
    return (
        np.asarray([float(value) for value in e_values]),
        np.asarray([float(value) for value in E_values]),
    )


def coulomb_speed_moments(
    spherical_order: int,
    spherical_radial_order: int,
    speed_power_order: int,
    mass_ratio: float,
    temperature_ratio: float,
    *,
    collision_frequency: float = 1.0,
    digits: int = 80,
) -> tuple[float, float]:
    r"""Return velocity-integrated Coulomb test and field speed functions.

    The result is ``(nu_T, nu_F)`` from equations (A5) and (A13) of Frei et
    al. (2021) for ``sigma=mass_ratio`` and ``tau=temperature_ratio``.
    """

    if any(
        order < 0
        for order in (spherical_order, spherical_radial_order, speed_power_order)
    ):
        raise ValueError("basis orders must be >= 0")
    if mass_ratio <= 0.0 or not math.isfinite(mass_ratio):
        raise ValueError("mass_ratio must be finite and > 0")
    if temperature_ratio <= 0.0 or not math.isfinite(temperature_ratio):
        raise ValueError("temperature_ratio must be finite and > 0")
    if collision_frequency < 0.0 or not math.isfinite(collision_frequency):
        raise ValueError("collision_frequency must be finite and >= 0")
    if digits < 16:
        raise ValueError("digits must be >= 16")

    import mpmath as mp

    with mp.workdps(digits):
        test_moment, field_moment = _coulomb_speed_moments_mp(
            spherical_order,
            spherical_radial_order,
            speed_power_order,
            mass_ratio,
            temperature_ratio,
            collision_frequency,
            mp,
        )
    return float(test_moment), float(field_moment)


def _coulomb_speed_moments_mp(
    spherical_order: int,
    spherical_radial_order: int,
    speed_power_order: int,
    mass_ratio: Any,
    temperature_ratio: Any,
    collision_frequency: Any,
    mp: Any,
    *,
    coulomb_e: Callable[[int], Any] | None = None,
    coulomb_E: Callable[[int], Any] | None = None,
) -> tuple[Any, Any]:
    p = spherical_order
    j = spherical_radial_order
    d = speed_power_order
    sigma = mp.mpf(mass_ratio)
    tau = mp.mpf(temperature_ratio)
    nu = mp.mpf(collision_frequency)
    chi = mp.sqrt(tau / sigma)
    e_integral = (
        (lambda order: _coulomb_e_mp(order, chi, mp))
        if coulomb_e is None
        else coulomb_e
    )
    E_integral = (
        (lambda order: _coulomb_E_mp(order, chi, mp))
        if coulomb_E is None
        else coulomb_E
    )
    inverse_sqrt_pi = 1 / mp.sqrt(mp.pi)
    test_moment = mp.mpf(0)
    field_moment = mp.mpf(0)

    for monomial_order in range(j + 1):
        laguerre_coefficient = _associated_laguerre_monomial_coefficient_mp(
            j,
            p,
            monomial_order,
            mp,
        )
        common_polynomial = (
            4 * monomial_order**2
            + 4 * monomial_order * (p - 1)
            + mp.mpf("1.5") * p * (p - 1)
        )
        test_erf_coefficients = (
            -4 * sigma * (tau - 1) / tau,
            4 * sigma / tau * monomial_order * (tau - 2)
            - p * (p - 2 * sigma + 4 * sigma / tau + 1),
            sigma / tau * common_polynomial,
        )
        test_gaussian_coefficients = (
            4 * (tau - 1) * (sigma + tau) / tau,
            -2 * (p + 2 * monomial_order) * (sigma / tau * (tau - 2) - 1),
            -sigma / tau * common_polynomial,
        )
        for summation_order in range(3):
            erf_coefficient = test_erf_coefficients[summation_order]
            gaussian_coefficient = test_gaussian_coefficients[summation_order]
            if erf_coefficient != 0:
                integral_order = d + monomial_order + p - summation_order
                test_moment += (
                    laguerre_coefficient
                    * erf_coefficient
                    * 4
                    * nu
                    * inverse_sqrt_pi
                    * E_integral(integral_order)
                )
            if gaussian_coefficient != 0:
                integral_order = d + monomial_order + p - summation_order + 1
                test_moment += (
                    laguerre_coefficient
                    * gaussian_coefficient
                    * 4
                    * chi
                    * nu
                    * inverse_sqrt_pi
                    * e_integral(integral_order)
                )

        field_prime_coefficients = (
            4 * tau**2 / sigma,
            -8 * (p * p + p - 1) / ((2 * p - 1) * (2 * p + 3)),
        )
        field_plus_coefficients = (
            -8 * p * (p - 1) * (monomial_order + 1) / ((2 * p + 1) * (2 * p - 1))
            - 8 * tau * (1 + p - sigma * p) / (sigma * (2 * p + 1)),
            8 * (p + 1) * (p + 2) / ((2 * p + 1) * (2 * p + 3)),
        )
        field_minus_coefficients = (
            4
            * (p + 1)
            * (p + 2)
            * (2 * p + 2 * monomial_order + 3)
            / ((2 * p + 1) * (2 * p + 3))
            + 8 * chi**2 * (p - sigma * p - sigma) / (2 * p + 1),
            -8 * p * (p - 1) / ((2 * p + 1) * (2 * p - 1)),
        )
        for summation_order in range(2):
            beta_e = (
                4
                * nu
                * inverse_sqrt_pi
                * chi ** (p + 2 * monomial_order + 2 * summation_order - 1)
                * e_integral(p + 1 + d + monomial_order + summation_order)
            )
            beta_plus = (
                4
                * nu
                * inverse_sqrt_pi
                * chi ** (p + 2 * summation_order - 1)
                * sum(
                    chi ** (2 * inner_order)
                    * mp.factorial(monomial_order)
                    / mp.factorial(inner_order)
                    * e_integral(p + summation_order + 1 + d + inner_order)
                    / 2
                    for inner_order in range(monomial_order + 1)
                )
            )
            gamma_factor = mp.gamma(p + monomial_order + mp.mpf("1.5"))
            beta_minus = (
                4
                * nu
                * inverse_sqrt_pi
                * chi ** (2 * summation_order - p - 2)
                * (
                    gamma_factor
                    / mp.gamma(mp.mpf("1.5"))
                    * E_integral(d + summation_order)
                    / 2
                    - sum(
                        chi ** (2 * inner_order + 1)
                        * gamma_factor
                        / mp.gamma(inner_order + mp.mpf("1.5"))
                        * e_integral(d + summation_order + inner_order + 1)
                        / 2
                        for inner_order in range(p + monomial_order + 1)
                    )
                )
            )
            field_moment += laguerre_coefficient * (
                field_prime_coefficients[summation_order] * beta_e
                + field_plus_coefficients[summation_order] * beta_plus
                + field_minus_coefficients[summation_order] * beta_minus
            )
    return test_moment, field_moment


def associated_laguerre_monomial_coefficients(
    polynomial_order: int,
    tensor_order: int,
    *,
    digits: int = 80,
) -> np.ndarray:
    r"""Return monomial coefficients of ``L_j^(p+1/2)``.

    This is equation (3.10) of Frei et al. (2021), evaluated with
    multiprecision arithmetic for the offline collision-table generator.
    Coefficient index ``ell`` multiplies ``x**ell``.
    """

    if polynomial_order < 0:
        raise ValueError("polynomial_order must be >= 0")
    if tensor_order < 0:
        raise ValueError("tensor_order must be >= 0")
    if digits < 16:
        raise ValueError("digits must be >= 16")

    import mpmath as mp

    with mp.workdps(digits):
        coefficients = [
            _associated_laguerre_monomial_coefficient_mp(
                polynomial_order,
                tensor_order,
                ell,
                mp,
            )
            for ell in range(polynomial_order + 1)
        ]
    return np.asarray([float(value) for value in coefficients])


def _associated_laguerre_monomial_coefficient_mp(
    polynomial_order: int,
    tensor_order: Any,
    monomial_order: int,
    mp: Any,
) -> Any:
    half = mp.mpf("0.5")
    return (
        (-1) ** monomial_order
        * mp.gamma(tensor_order + polynomial_order + 1 + half)
        / (
            mp.factorial(polynomial_order - monomial_order)
            * mp.gamma(monomial_order + tensor_order + 1 + half)
            * mp.factorial(monomial_order)
        )
    )


def _legendre_monomial_coefficient_mp(order: int, power: int, mp: Any) -> Any:
    if power < 0 or power > order or (order - power) % 2:
        return mp.mpf(0)
    half_difference = (order - power) // 2
    return (
        (-1) ** half_difference
        * mp.factorial(order + power)
        / (
            mp.power(2, order)
            * mp.factorial(half_difference)
            * mp.factorial((order + power) // 2)
            * mp.factorial(power)
        )
    )


def _spherical_polynomial_mp(
    legendre_order: int,
    radial_order: int,
    mp: Any,
    *,
    associated_laguerre: Callable[[int, Any, int], Any] | None = None,
    legendre_monomial: Callable[[int, int], Any] | None = None,
) -> tuple[tuple[int, int, Any], ...]:
    """Expand one spherical basis polynomial in parallel/perpendicular powers."""

    if associated_laguerre is None:
        associated_laguerre = lambda n, alpha, power: (  # noqa: E731
            _associated_laguerre_monomial_coefficient_mp(n, alpha, power, mp)
        )
    if legendre_monomial is None:
        legendre_monomial = lambda n, power: _legendre_monomial_coefficient_mp(  # noqa: E731
            n, power, mp
        )
    coefficients: dict[tuple[int, int], Any] = {}
    for parallel_power in range(legendre_order + 1):
        legendre = legendre_monomial(legendre_order, parallel_power)
        if legendre == 0:
            continue
        for radial_power in range(radial_order + 1):
            laguerre = associated_laguerre(
                radial_order,
                legendre_order,
                radial_power,
            )
            total_radial_power = (legendre_order - parallel_power) // 2 + radial_power
            for parallel_radial_power in range(total_radial_power + 1):
                key = (
                    parallel_power + 2 * parallel_radial_power,
                    total_radial_power - parallel_radial_power,
                )
                coefficients[key] = coefficients.get(key, mp.mpf(0)) + (
                    legendre
                    * laguerre
                    * math.comb(total_radial_power, parallel_radial_power)
                )
    return tuple(
        (parallel_power, perpendicular_power, coefficient)
        for (parallel_power, perpendicular_power), coefficient in sorted(
            coefficients.items()
        )
        if coefficient != 0
    )


def _hermite_gaussian_moment_mp(hermite_order: int, power: int, mp: Any) -> Any:
    """Integrate ``exp(-x**2) H_n(x) x**power`` analytically."""

    total = mp.mpf(0)
    for pair_order in range(hermite_order // 2 + 1):
        combined_power = power + hermite_order - 2 * pair_order
        if combined_power % 2:
            continue
        coefficient = (
            (-1) ** pair_order
            * mp.factorial(hermite_order)
            * mp.power(2, hermite_order - 2 * pair_order)
            / (mp.factorial(pair_order) * mp.factorial(hermite_order - 2 * pair_order))
        )
        total += coefficient * mp.gamma((combined_power + 1) / 2)
    return total


def _laguerre_exponential_moment_mp(
    laguerre_order: int,
    power: int,
    mp: Any,
    *,
    associated_laguerre: Callable[[int, Any, int], Any] | None = None,
) -> Any:
    """Integrate ``exp(-x) L_n(x) x**power`` analytically."""

    if associated_laguerre is None:
        associated_laguerre = lambda n, alpha, monomial: (  # noqa: E731
            _associated_laguerre_monomial_coefficient_mp(n, alpha, monomial, mp)
        )
    return sum(
        associated_laguerre(laguerre_order, -mp.mpf("0.5"), monomial_order)
        * mp.factorial(power + monomial_order)
        for monomial_order in range(laguerre_order + 1)
    )


def _legendre_to_hermite_laguerre_mp(
    legendre_order: int,
    radial_order: int,
    hermite_order: int,
    laguerre_order: int,
    mp: Any,
    *,
    spherical_polynomial: Callable[[int, int], tuple[tuple[int, int, Any], ...]]
    | None = None,
    hermite_gaussian_moment: Callable[[int, int], Any] | None = None,
    laguerre_exponential_moment: Callable[[int, int], Any] | None = None,
) -> Any:
    if spherical_polynomial is None:
        spherical_polynomial = lambda p, j: _spherical_polynomial_mp(  # noqa: E731
            p, j, mp
        )
    if hermite_gaussian_moment is None:
        hermite_gaussian_moment = lambda g, power: (  # noqa: E731
            _hermite_gaussian_moment_mp(g, power, mp)
        )
    if laguerre_exponential_moment is None:
        laguerre_exponential_moment = lambda h, power: (  # noqa: E731
            _laguerre_exponential_moment_mp(h, power, mp)
        )
    projection = sum(
        coefficient
        * hermite_gaussian_moment(hermite_order, parallel_power)
        * laguerre_exponential_moment(laguerre_order, perpendicular_power)
        for parallel_power, perpendicular_power, coefficient in spherical_polynomial(
            legendre_order, radial_order
        )
    )
    return projection / (
        mp.sqrt(mp.pi) * mp.power(2, hermite_order) * mp.factorial(hermite_order)
    )


def legendre_to_hermite_laguerre_coefficient(
    legendre_order: int,
    radial_order: int,
    hermite_order: int,
    laguerre_order: int,
    *,
    digits: int = 80,
) -> float:
    r"""Return the isotropic Legendre-to-Hermite--Laguerre coefficient.

    This implements equation (A4) of Jorge, Ricci & Loureiro (2017).  The
    coefficient maps ``c**l P_l(xi) L_k^(l+1/2)(c**2)`` onto
    ``H_p(s_parallel) L_j(s_perp**2)``.  Coefficients on different total-degree
    shells are exactly zero.
    """

    indices = (legendre_order, radial_order, hermite_order, laguerre_order)
    if any(index < 0 for index in indices):
        raise ValueError("basis orders must be >= 0")
    if digits < 16:
        raise ValueError("digits must be >= 16")
    if hermite_order + 2 * laguerre_order != legendre_order + 2 * radial_order:
        return 0.0

    import mpmath as mp

    with mp.workdps(digits):
        total = _legendre_to_hermite_laguerre_mp(
            legendre_order,
            radial_order,
            hermite_order,
            laguerre_order,
            mp,
        )
    return float(total)


def hermite_laguerre_to_legendre_coefficient(
    hermite_order: int,
    laguerre_order: int,
    legendre_order: int,
    radial_order: int,
    *,
    digits: int = 80,
) -> float:
    r"""Return the inverse isotropic basis-transform coefficient.

    This applies equation (A3) of Jorge, Ricci & Loureiro (2017) to the
    multiprecision forward coefficient.
    """

    indices = (legendre_order, radial_order, hermite_order, laguerre_order)
    if any(index < 0 for index in indices):
        raise ValueError("basis orders must be >= 0")
    if digits < 16:
        raise ValueError("digits must be >= 16")
    if hermite_order + 2 * laguerre_order != legendre_order + 2 * radial_order:
        return 0.0

    import mpmath as mp

    with mp.workdps(digits):
        forward = _legendre_to_hermite_laguerre_mp(
            legendre_order,
            radial_order,
            hermite_order,
            laguerre_order,
            mp,
        )
        factor = (
            mp.sqrt(mp.pi)
            * mp.power(2, hermite_order)
            * mp.factorial(hermite_order)
            * (legendre_order + mp.mpf("0.5"))
            * mp.factorial(radial_order)
            / mp.gamma(radial_order + legendre_order + mp.mpf("1.5"))
        )
        inverse = factor * forward
    return float(inverse)


def _associated_legendre_radial_overlap_mp(
    legendre_order: int,
    radial_order: int,
    bessel_order: int,
    auxiliary_legendre_order: int,
    auxiliary_radial_order: int,
    mp: Any,
    *,
    associated_laguerre: Callable[[int, Any, int], Any],
) -> Any:
    """Contract the radial factor of one associated-basis overlap."""

    left_radial = tuple(
        associated_laguerre(radial_order, legendre_order, power)
        for power in range(radial_order + 1)
    )
    auxiliary_radial = tuple(
        associated_laguerre(
            auxiliary_radial_order,
            auxiliary_legendre_order,
            power,
        )
        for power in range(auxiliary_radial_order + 1)
    )
    radial_convolution = [mp.mpf(0)] * (len(left_radial) + len(auxiliary_radial) - 1)
    for left_power, left_value in enumerate(left_radial):
        for auxiliary_power, auxiliary_value in enumerate(auxiliary_radial):
            radial_convolution[left_power + auxiliary_power] += (
                left_value * auxiliary_value
            )
    radial_offset = (
        mp.mpf(legendre_order + auxiliary_legendre_order - bessel_order + 3) / 2
    )
    return sum(
        value * mp.gamma(power + radial_offset)
        for power, value in enumerate(radial_convolution)
    )


def _associated_legendre_angular_overlap_mp(
    legendre_order: int,
    bessel_order: int,
    auxiliary_legendre_order: int,
    mp: Any,
    *,
    legendre_monomial: Callable[[int, int], Any],
) -> Any:
    """Contract the angular factor shared by all radial-order pairs."""

    left_angular = []
    for left_power in range(bessel_order, legendre_order + 1):
        left_legendre = legendre_monomial(legendre_order, left_power)
        left_angular.append(
            left_legendre
            * mp.factorial(left_power)
            / mp.factorial(left_power - bessel_order)
        )
    auxiliary_angular = tuple(
        legendre_monomial(auxiliary_legendre_order, power)
        for power in range(auxiliary_legendre_order + 1)
    )
    angular_convolution = [mp.mpf(0)] * (len(left_angular) + len(auxiliary_angular) - 1)
    for left_power, left_value in enumerate(left_angular):
        for auxiliary_power, auxiliary_value in enumerate(auxiliary_angular):
            angular_convolution[left_power + auxiliary_power] += (
                left_value * auxiliary_value
            )
    return sum(
        value * (1 + (-1) ** power) / (2 * (power + 1))
        for power, value in enumerate(angular_convolution)
    )


def _associated_legendre_overlap_mp(
    legendre_order: int,
    radial_order: int,
    bessel_order: int,
    auxiliary_legendre_order: int,
    auxiliary_radial_order: int,
    mp: Any,
    *,
    associated_laguerre: Callable[[int, Any, int], Any],
    legendre_monomial: Callable[[int, int], Any],
) -> Any:
    """Contract one associated-basis overlap shared by a parity shell."""

    radial = _associated_legendre_radial_overlap_mp(
        legendre_order,
        radial_order,
        bessel_order,
        auxiliary_legendre_order,
        auxiliary_radial_order,
        mp,
        associated_laguerre=associated_laguerre,
    )
    angular = _associated_legendre_angular_overlap_mp(
        legendre_order,
        bessel_order,
        auxiliary_legendre_order,
        mp,
        legendre_monomial=legendre_monomial,
    )
    return radial * angular


def _associated_spherical_polynomial_mp(
    legendre_order: int,
    radial_order: int,
    bessel_order: int,
    mp: Any,
    *,
    associated_laguerre: Callable[[int, Any, int], Any] | None = None,
    legendre_monomial: Callable[[int, int], Any] | None = None,
) -> tuple[tuple[int, int, Any], ...]:
    """Expand the finite-m spherical basis after factoring ``s_perp**m``."""

    if associated_laguerre is None:
        associated_laguerre = lambda n, alpha, power: (  # noqa: E731
            _associated_laguerre_monomial_coefficient_mp(n, alpha, power, mp)
        )
    if legendre_monomial is None:
        legendre_monomial = lambda n, power: _legendre_monomial_coefficient_mp(  # noqa: E731
            n, power, mp
        )
    coefficients: dict[tuple[int, int], Any] = {}
    for legendre_power in range(bessel_order, legendre_order + 1):
        legendre = legendre_monomial(legendre_order, legendre_power)
        if legendre == 0:
            continue
        derivative = (
            (-1) ** bessel_order
            * legendre
            * mp.factorial(legendre_power)
            / mp.factorial(legendre_power - bessel_order)
        )
        for radial_power in range(radial_order + 1):
            total_radial_power = (legendre_order - legendre_power) // 2 + radial_power
            common = derivative * associated_laguerre(
                radial_order,
                legendre_order,
                radial_power,
            )
            for parallel_radial_power in range(total_radial_power + 1):
                key = (
                    legendre_power - bessel_order + 2 * parallel_radial_power,
                    total_radial_power - parallel_radial_power,
                )
                coefficients[key] = coefficients.get(key, mp.mpf(0)) + (
                    common * math.comb(total_radial_power, parallel_radial_power)
                )
    return tuple(
        (parallel_power, perpendicular_power, coefficient)
        for (parallel_power, perpendicular_power), coefficient in sorted(
            coefficients.items()
        )
        if coefficient != 0
    )


def _associated_legendre_to_hermite_laguerre_overlap_mp(
    legendre_order: int,
    radial_order: int,
    bessel_order: int,
    hermite_order: int,
    laguerre_order: int,
    mp: Any,
    *,
    base_transform: Callable[[int, int, int, int], Any] | None = None,
    associated_laguerre: Callable[[int, Any, int], Any] | None = None,
    legendre_monomial: Callable[[int, int], Any] | None = None,
    associated_overlap: Callable[[int, int, int, int, int], Any] | None = None,
) -> Any:
    left_degree = legendre_order + 2 * radial_order - bessel_order
    right_degree = hermite_order + 2 * laguerre_order
    if right_degree > left_degree or (left_degree - right_degree) % 2:
        return mp.mpf(0)

    half = mp.mpf("0.5")
    total = mp.mpf(0)
    if base_transform is None:

        def base_transform(p: int, j: int, g: int, s: int) -> Any:
            return _legendre_to_hermite_laguerre_mp(p, j, g, s, mp)

    if associated_laguerre is None:

        def associated_laguerre(n: int, alpha: Any, power: int) -> Any:
            return _associated_laguerre_monomial_coefficient_mp(n, alpha, power, mp)

    if legendre_monomial is None:

        def legendre_monomial(n: int, power: int) -> Any:
            return _legendre_monomial_coefficient_mp(n, power, mp)

    if associated_overlap is None:

        def associated_overlap(
            p: int,
            j: int,
            m: int,
            auxiliary_p: int,
            auxiliary_j: int,
        ) -> Any:
            return _associated_legendre_overlap_mp(
                p,
                j,
                m,
                auxiliary_p,
                auxiliary_j,
                mp,
                associated_laguerre=associated_laguerre,
                legendre_monomial=legendre_monomial,
            )

    for auxiliary_radial_order in range(right_degree // 2 + 1):
        auxiliary_legendre_order = right_degree - 2 * auxiliary_radial_order
        transform = base_transform(
            auxiliary_legendre_order,
            auxiliary_radial_order,
            hermite_order,
            laguerre_order,
        )
        prefactor = (
            transform
            * (auxiliary_legendre_order + half)
            * mp.factorial(auxiliary_radial_order)
            / mp.gamma(
                auxiliary_legendre_order + auxiliary_radial_order + mp.mpf("1.5")
            )
        )
        inner = associated_overlap(
            legendre_order,
            radial_order,
            bessel_order,
            auxiliary_legendre_order,
            auxiliary_radial_order,
        )
        total += prefactor * inner

    # Equation (B5) is half-normalized at m=0 and has the opposite odd-m
    # phase under scipy's associated-Legendre convention.  This factor is
    # fixed by the m=0 endpoint and independent velocity-space projection.
    return 2 * (-1) ** bessel_order * total


def _associated_legendre_to_hermite_laguerre_mp(
    legendre_order: int,
    radial_order: int,
    bessel_order: int,
    hermite_order: int,
    laguerre_order: int,
    mp: Any,
    *,
    associated_spherical_polynomial: Callable[
        [int, int, int], tuple[tuple[int, int, Any], ...]
    ]
    | None = None,
    hermite_gaussian_moment: Callable[[int, int], Any] | None = None,
    laguerre_exponential_moment: Callable[[int, int], Any] | None = None,
) -> Any:
    """Transform the factored finite-m basis by exact polynomial projection."""

    left_degree = legendre_order + 2 * radial_order - bessel_order
    right_degree = hermite_order + 2 * laguerre_order
    if right_degree > left_degree or (left_degree - right_degree) % 2:
        return mp.mpf(0)
    if associated_spherical_polynomial is None:
        associated_spherical_polynomial = lambda p, j, m: (  # noqa: E731
            _associated_spherical_polynomial_mp(p, j, m, mp)
        )
    if hermite_gaussian_moment is None:
        hermite_gaussian_moment = lambda g, power: (  # noqa: E731
            _hermite_gaussian_moment_mp(g, power, mp)
        )
    if laguerre_exponential_moment is None:
        laguerre_exponential_moment = lambda h, power: (  # noqa: E731
            _laguerre_exponential_moment_mp(h, power, mp)
        )
    projection = sum(
        coefficient
        * hermite_gaussian_moment(hermite_order, parallel_power)
        * laguerre_exponential_moment(laguerre_order, perpendicular_power)
        for parallel_power, perpendicular_power, coefficient in associated_spherical_polynomial(
            legendre_order,
            radial_order,
            bessel_order,
        )
    )
    return projection / (
        mp.sqrt(mp.pi) * mp.power(2, hermite_order) * mp.factorial(hermite_order)
    )


def associated_legendre_to_hermite_laguerre_coefficient(
    legendre_order: int,
    radial_order: int,
    bessel_order: int,
    hermite_order: int,
    laguerre_order: int,
    *,
    digits: int = 80,
) -> float:
    r"""Return a finite-Bessel-order collision-basis coefficient.

    This is the convention-corrected forward transform from equation (B5) of
    Jorge, Frei & Ricci (2019).  Unlike the isotropic transform, finite
    ``bessel_order`` couples every lower reduced-degree shell of the same
    parity.
    """

    indices = (
        legendre_order,
        radial_order,
        bessel_order,
        hermite_order,
        laguerre_order,
    )
    if any(index < 0 for index in indices):
        raise ValueError("basis orders must be >= 0")
    if bessel_order > legendre_order:
        raise ValueError("bessel_order must be <= legendre_order")
    if digits < 16:
        raise ValueError("digits must be >= 16")

    import mpmath as mp

    with mp.workdps(digits):
        coefficient = _associated_legendre_to_hermite_laguerre_mp(
            legendre_order,
            radial_order,
            bessel_order,
            hermite_order,
            laguerre_order,
            mp,
        )
    return float(coefficient)


def associated_basis_transform_matrices(
    bessel_order: int,
    maximum_reduced_degree: int,
    *,
    digits: int = 80,
) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
    np.ndarray,
    np.ndarray,
]:
    r"""Generate and invert one finite-``m`` parity block.

    The block contains all nonnegative reduced-degree shells with the parity
    of ``maximum_reduced_degree``.  Rows are Hermite--Laguerre ``(p, j)``
    orders, columns are associated Legendre--Laguerre ``(l, k)`` orders, and
    both order tuples are sorted by increasing reduced degree.  The inverse is
    formed from the complete multiprecision block rather than equation (B6),
    whose literal finite-``m`` normalization fails the inverse identity.
    """

    if bessel_order < 0:
        raise ValueError("bessel_order must be >= 0")
    if maximum_reduced_degree < 0:
        raise ValueError("maximum_reduced_degree must be >= 0")
    if digits < 16:
        raise ValueError("digits must be >= 16")

    parity = maximum_reduced_degree % 2
    right_orders = tuple(
        (degree - 2 * radial_order, radial_order)
        for degree in range(parity, maximum_reduced_degree + 1, 2)
        for radial_order in range(degree // 2 + 1)
    )
    left_orders = tuple(
        (bessel_order + degree - 2 * radial_order, radial_order)
        for degree in range(parity, maximum_reduced_degree + 1, 2)
        for radial_order in range(degree // 2 + 1)
    )

    import mpmath as mp

    with mp.workdps(digits):
        forward_mp = mp.matrix(
            [
                [
                    _associated_legendre_to_hermite_laguerre_mp(
                        legendre_order,
                        radial_order,
                        bessel_order,
                        hermite_order,
                        laguerre_order,
                        mp,
                    )
                    for legendre_order, radial_order in left_orders
                ]
                for hermite_order, laguerre_order in right_orders
            ]
        )
        inverse_mp = forward_mp**-1
        forward = np.asarray(forward_mp.tolist(), dtype=np.float64)
        inverse = np.asarray(inverse_mp.tolist(), dtype=np.float64)
    return right_orders, left_orders, forward, inverse


def _laguerre_product_expansion_coefficient_mp(
    associated_order: int,
    associated_polynomial_order: int,
    laguerre_order: int,
    output_order: int,
    radial_power: int,
    mp: Any,
    *,
    monomial_coefficient: Callable[[int, Any, int], Any] | None = None,
    laguerre_moment: Callable[[int, int], Any] | None = None,
) -> Any:
    if output_order > associated_polynomial_order + laguerre_order + radial_power:
        return mp.mpf(0)

    half = mp.mpf("0.5")
    if monomial_coefficient is None:

        def monomial_coefficient(order: int, alpha: Any, power: int) -> Any:
            return _associated_laguerre_monomial_coefficient_mp(order, alpha, power, mp)

    associated = tuple(
        monomial_coefficient(
            associated_polynomial_order,
            associated_order - half,
            power,
        )
        for power in range(associated_polynomial_order + 1)
    )
    ordinary = tuple(
        monomial_coefficient(laguerre_order, -half, power)
        for power in range(laguerre_order + 1)
    )
    convolution = [mp.mpf(0)] * (len(associated) + len(ordinary) - 1)
    for associated_power, associated_value in enumerate(associated):
        for laguerre_power, laguerre_value in enumerate(ordinary):
            convolution[associated_power + laguerre_power] += (
                associated_value * laguerre_value
            )
    if laguerre_moment is None:

        def laguerre_moment(order: int, power: int) -> Any:
            return sum(
                monomial_coefficient(order, -half, monomial_order)
                * mp.factorial(power + monomial_order)
                for monomial_order in range(order + 1)
            )

    return sum(
        product_value * laguerre_moment(output_order, product_power + radial_power)
        for product_power, product_value in enumerate(convolution)
    )


def laguerre_product_expansion_coefficient(
    associated_order: int,
    associated_polynomial_order: int,
    laguerre_order: int,
    output_order: int,
    *,
    radial_power: int = 0,
    digits: int = 80,
) -> float:
    r"""Expand ``x**r L_n^m(x) L_k(x)`` in ordinary Laguerre modes.

    ``radial_power=associated_order`` implements equations (3.36)--(3.37) of
    Frei et al. (2021); ``radial_power=0`` implements equations
    (3.44)--(3.45).  The returned coefficient multiplies ``L_output_order``.
    """

    indices = (
        associated_order,
        associated_polynomial_order,
        laguerre_order,
        output_order,
        radial_power,
    )
    if any(index < 0 for index in indices):
        raise ValueError("Laguerre orders and radial_power must be >= 0")
    if digits < 16:
        raise ValueError("digits must be >= 16")

    import mpmath as mp

    with mp.workdps(digits):
        coefficient = _laguerre_product_expansion_coefficient_mp(
            associated_order,
            associated_polynomial_order,
            laguerre_order,
            output_order,
            radial_power,
            mp,
        )
    return float(coefficient)


def _hermite_laguerre_to_associated_legendre_mp(
    hermite_order: int,
    laguerre_order: int,
    spherical_order: int,
    spherical_radial_order: int,
    bessel_order: int,
    mp: Any,
    *,
    associated_spherical_polynomial: Callable[
        [int, int, int], tuple[tuple[int, int, Any], ...]
    ]
    | None = None,
    hermite_gaussian_moment: Callable[[int, int], Any] | None = None,
    laguerre_exponential_moment: Callable[[int, int], Any] | None = None,
) -> Any:
    if hermite_order > spherical_order + bessel_order + 2 * spherical_radial_order:
        return mp.mpf(0)
    if (hermite_order - spherical_order - bessel_order) % 2:
        return mp.mpf(0)

    # The Hermite normalization in equation (3.33) cancels exactly between
    # the inverse-basis prefactor and the velocity-space projection.
    prefactor = (
        mp.factorial(spherical_radial_order)
        * (spherical_order + mp.mpf("0.5"))
        / mp.gamma(spherical_radial_order + spherical_order + mp.mpf("1.5"))
        * mp.factorial(spherical_order - bessel_order)
        / mp.factorial(spherical_order + bessel_order)
    )
    if associated_spherical_polynomial is None:
        associated_spherical_polynomial = lambda p, j, m: (  # noqa: E731
            _associated_spherical_polynomial_mp(p, j, m, mp)
        )
    if hermite_gaussian_moment is None:
        hermite_gaussian_moment = lambda g, power: (  # noqa: E731
            _hermite_gaussian_moment_mp(g, power, mp)
        )
    if laguerre_exponential_moment is None:
        laguerre_exponential_moment = lambda k, power: (  # noqa: E731
            _laguerre_exponential_moment_mp(k, power, mp)
        )

    polynomial = associated_spherical_polynomial(
        spherical_order,
        spherical_radial_order,
        bessel_order,
    )
    if not polynomial:
        return mp.mpf(0)
    if hermite_order > max(term[0] for term in polynomial):
        return mp.mpf(0)
    if laguerre_order > max(term[1] + bessel_order for term in polynomial):
        return mp.mpf(0)

    # Equation (3.33) expands x**m L_k in ordinary Laguerre modes and then
    # projects each mode.  Laguerre orthogonality collapses that sum to this
    # single weighted projection of the factored spherical polynomial.
    contraction = sum(
        coefficient
        * hermite_gaussian_moment(hermite_order, parallel_power)
        * laguerre_exponential_moment(
            laguerre_order,
            perpendicular_power + bessel_order,
        )
        for parallel_power, perpendicular_power, coefficient in polynomial
    )
    return prefactor * contraction


def hermite_laguerre_to_associated_legendre_coefficient(
    hermite_order: int,
    laguerre_order: int,
    spherical_order: int,
    spherical_radial_order: int,
    bessel_order: int,
    *,
    digits: int = 80,
) -> float:
    r"""Return the finite-``m`` inverse collision-basis coefficient.

    This implements equation (3.33) of Frei et al. (2021), including its
    weighted Laguerre-product contraction.  It maps ``H_l L_k x**(m/2)`` onto
    ``s**p P_p^m L_j^(p+1/2)``.
    """

    indices = (
        hermite_order,
        laguerre_order,
        spherical_order,
        spherical_radial_order,
        bessel_order,
    )
    if any(index < 0 for index in indices):
        raise ValueError("basis orders must be >= 0")
    if bessel_order > spherical_order:
        raise ValueError("bessel_order must be <= spherical_order")
    if digits < 16:
        raise ValueError("digits must be >= 16")

    import mpmath as mp

    with mp.workdps(digits):
        coefficient = _hermite_laguerre_to_associated_legendre_mp(
            hermite_order,
            laguerre_order,
            spherical_order,
            spherical_radial_order,
            bessel_order,
            mp,
        )
    return float(coefficient)


def gyroaveraged_spherical_moment_coefficient(
    spherical_order: int,
    spherical_radial_order: int,
    bessel_order: int,
    hermite_order: int,
    laguerre_order: int,
    bessel_argument: float,
    *,
    maximum_bessel_laguerre_order: int = 24,
    digits: int = 80,
) -> float:
    r"""Map one gyro-moment into a finite-``m`` spherical particle moment.

    This evaluates one ``N^(g,s)`` coefficient in equation (3.35) of Frei et
    al. (2021).  The Bessel--Laguerre sum is explicitly truncated at
    ``maximum_bessel_laguerre_order``; convergence must be checked when tables
    are generated.
    """

    indices = (
        spherical_order,
        spherical_radial_order,
        bessel_order,
        hermite_order,
        laguerre_order,
        maximum_bessel_laguerre_order,
    )
    if any(index < 0 for index in indices):
        raise ValueError("basis and truncation orders must be >= 0")
    if bessel_order > spherical_order:
        raise ValueError("bessel_order must be <= spherical_order")
    if not math.isfinite(bessel_argument) or bessel_argument < 0.0:
        raise ValueError("bessel_argument must be finite and >= 0")
    if digits < 16:
        raise ValueError("digits must be >= 16")

    import mpmath as mp

    with mp.workdps(digits):
        coefficient = _gyroaveraged_spherical_moment_coefficient_mp(
            spherical_order,
            spherical_radial_order,
            bessel_order,
            hermite_order,
            laguerre_order,
            bessel_argument,
            maximum_bessel_laguerre_order,
            mp,
        )
    return float(coefficient)


def _bessel_laguerre_kernels_mp(
    bessel_argument: Any,
    bessel_order: int,
    count: int,
    mp: Any,
) -> tuple[Any, ...]:
    """Return the finite-m Poisson weights shared by collision projections."""

    half_argument = mp.mpf(bessel_argument) / 2
    radial_argument = half_argument**2
    exponential = mp.exp(-radial_argument)
    return tuple(
        exponential
        * radial_argument**order
        * half_argument**bessel_order
        / mp.factorial(order + bessel_order)
        for order in range(count)
    )


def _radial_poisson_kernels_mp(
    bessel_argument: Any,
    count: int,
    mp: Any,
) -> tuple[Any, ...]:
    """Return radial Poisson weights used by the polarization contraction."""

    radial_argument = (mp.mpf(bessel_argument) / 2) ** 2
    exponential = mp.exp(-radial_argument)
    return tuple(
        exponential * radial_argument**order / mp.factorial(order)
        for order in range(count)
    )


def _bessel_weighted_laguerre_products_mp(
    bessel_order: int,
    output_laguerre: int,
    expansion_orders: tuple[int, ...],
    bessel_weights: tuple[Any, ...],
    laguerre_product: Callable[[int, int, int, int], Any],
    mp: Any,
) -> tuple[Any, ...]:
    """Contract the Bessel index before applying inverse basis transforms."""

    maximum_product = output_laguerre + max(expansion_orders)
    weights = [mp.mpf(0)] * (maximum_product + 1)
    for expansion_order, bessel_weight in zip(
        expansion_orders, bessel_weights, strict=True
    ):
        if bessel_weight == 0:
            continue
        for product_order in range(output_laguerre + expansion_order + 1):
            weights[product_order] += bessel_weight * laguerre_product(
                bessel_order,
                expansion_order,
                output_laguerre,
                product_order,
            )
    return tuple(weights)


def _forked_ordered_map(
    builder: Callable[[int], Any],
    item_count: int,
    worker_count: int,
) -> list[Any]:
    """Evaluate disjoint offline rows with copy-on-write precomputed state."""

    if worker_count == 1:
        return [builder(index) for index in range(item_count)]

    import multiprocessing

    if "fork" not in multiprocessing.get_all_start_methods():
        raise RuntimeError("parallel collision generation requires POSIX fork")
    context = multiprocessing.get_context("fork")
    chunks = [
        tuple(range(worker, item_count, worker_count)) for worker in range(worker_count)
    ]
    readers = []
    processes = []

    def run_chunk(indices: tuple[int, ...], writer: Any) -> None:
        try:
            writer.send((True, [builder(index) for index in indices]))
        except Exception as error:  # pragma: no cover - child failure propagation
            writer.send((False, repr(error)))
        finally:
            writer.close()

    for chunk in chunks:
        reader, writer = context.Pipe(duplex=False)
        process = context.Process(target=run_chunk, args=(chunk, writer))
        process.start()
        writer.close()
        readers.append(reader)
        processes.append(process)

    results = []
    try:
        for reader in readers:
            passed, payload = reader.recv()
            if not passed:
                raise RuntimeError(f"parallel collision row failed: {payload}")
            results.extend(payload)
    finally:
        for reader in readers:
            reader.close()
        for process in processes:
            process.join()
            if process.is_alive():
                process.terminate()
    return sorted(results, key=lambda result: result[0])


def _resolve_collision_angular_orders(
    angular_limit: int,
    included_angular_orders: tuple[int, ...] | None,
    *,
    drift_kinetic: bool,
) -> tuple[int, ...]:
    """Return a validated, deterministic angular-harmonic subset."""

    if included_angular_orders is None:
        return (0,) if drift_kinetic else tuple(range(angular_limit + 1))
    orders = tuple(included_angular_orders)
    if not orders or orders != tuple(sorted(set(orders))):
        raise ValueError("included_angular_orders must be nonempty, unique, and sorted")
    if any(order < 0 or order > angular_limit for order in orders):
        raise ValueError("included_angular_orders exceed the angular truncation")
    if drift_kinetic and orders != (0,):
        raise ValueError("drift-kinetic generation supports only angular order zero")
    return orders


def _gyroaveraged_spherical_moment_coefficient_mp(
    spherical_order: int,
    spherical_radial_order: int,
    bessel_order: int,
    hermite_order: int,
    laguerre_order: int,
    bessel_argument: Any,
    maximum_bessel_laguerre_order: int,
    mp: Any,
    *,
    associated_transform: Callable[[int, int, int, int, int], Any] | None = None,
    laguerre_product: Callable[[int, int, int, int, int], Any] | None = None,
    bessel_kernels: tuple[Any, ...] | None = None,
    bessel_product_projection: Callable[[int, int], Any] | None = None,
    float64_final_contraction: bool = False,
) -> Any:
    b = mp.mpf(bessel_argument)
    coefficient = 0.0 if float64_final_contraction else mp.mpf(0)
    remaining_degree = (
        spherical_order + 2 * spherical_radial_order - bessel_order - hermite_order
    )
    if remaining_degree < 0 or remaining_degree % 2:
        return coefficient
    maximum_auxiliary_laguerre = remaining_degree // 2
    if associated_transform is None:

        def associated_transform(p: int, j: int, m: int, g: int, s: int) -> Any:
            return _associated_legendre_to_hermite_laguerre_mp(p, j, m, g, s, mp)

    if laguerre_product is None:

        def laguerre_product(m: int, n: int, k: int, output: int, radial: int) -> Any:
            return _laguerre_product_expansion_coefficient_mp(
                m, n, k, output, radial, mp
            )

    if bessel_kernels is None:
        bessel_kernels = _bessel_laguerre_kernels_mp(
            b,
            bessel_order,
            maximum_bessel_laguerre_order + 1,
            mp,
        )

    for auxiliary_laguerre_order in range(maximum_auxiliary_laguerre + 1):
        transform = associated_transform(
            spherical_order,
            spherical_radial_order,
            bessel_order,
            hermite_order,
            auxiliary_laguerre_order,
        )
        if transform == 0:
            continue
        if bessel_product_projection is not None:
            coefficient += float(transform) * float(
                bessel_product_projection(auxiliary_laguerre_order, laguerre_order)
            )
            continue
        for bessel_laguerre_order in range(maximum_bessel_laguerre_order + 1):
            product = laguerre_product(
                bessel_order,
                bessel_laguerre_order,
                auxiliary_laguerre_order,
                laguerre_order,
                bessel_order,
            )
            if product == 0:
                continue
            if float64_final_contraction:
                coefficient += (
                    float(transform)
                    * float(product)
                    * float(bessel_kernels[bessel_laguerre_order])
                )
            else:
                coefficient += (
                    transform * product * bessel_kernels[bessel_laguerre_order]
                )
    normalization = mp.sqrt(mp.power(2, hermite_order) * mp.factorial(hermite_order))
    coefficient *= float(normalization) if float64_final_contraction else normalization
    return coefficient


def _gyroaveraged_polarization_coefficient_mp(
    spherical_order: int,
    spherical_radial_order: int,
    bessel_order: int,
    bessel_argument: Any,
    maximum_bessel_laguerre_order: int,
    mp: Any,
    *,
    associated_transform: Callable[[int, int, int, int, int], Any] | None = None,
    laguerre_product: Callable[[int, int, int, int, int], Any] | None = None,
    bessel_kernels: tuple[Any, ...] | None = None,
    radial_kernels: tuple[Any, ...] | None = None,
    float64_final_contraction: bool = False,
) -> Any:
    b = mp.mpf(bessel_argument)
    total = 0.0 if float64_final_contraction else mp.mpf(0)
    remaining_degree = spherical_order + 2 * spherical_radial_order - bessel_order
    if remaining_degree % 2:
        return total
    maximum_auxiliary_laguerre = remaining_degree // 2
    kernel_count = (
        2 * maximum_bessel_laguerre_order
        + bessel_order
        + maximum_auxiliary_laguerre
        + 1
    )
    if radial_kernels is None:
        radial_kernels = _radial_poisson_kernels_mp(
            b,
            kernel_count,
            mp,
        )
    if bessel_kernels is None:
        bessel_kernels = _bessel_laguerre_kernels_mp(
            b,
            bessel_order,
            maximum_bessel_laguerre_order + 1,
            mp,
        )
    if associated_transform is None:

        def associated_transform(p: int, j: int, m: int, g: int, s: int) -> Any:
            return _associated_legendre_to_hermite_laguerre_mp(p, j, m, g, s, mp)

    if laguerre_product is None:

        def laguerre_product(m: int, n: int, k: int, output: int, radial: int) -> Any:
            return _laguerre_product_expansion_coefficient_mp(
                m, n, k, output, radial, mp
            )

    for auxiliary_laguerre_order in range(maximum_auxiliary_laguerre + 1):
        transform = associated_transform(
            spherical_order,
            spherical_radial_order,
            bessel_order,
            0,
            auxiliary_laguerre_order,
        )
        if transform == 0:
            continue
        for bessel_laguerre_order in range(maximum_bessel_laguerre_order + 1):
            leading = (
                float(transform) * float(bessel_kernels[bessel_laguerre_order])
                if float64_final_contraction
                else transform * bessel_kernels[bessel_laguerre_order]
            )
            if leading == 0:
                continue
            for output_order in range(
                bessel_laguerre_order + bessel_order + auxiliary_laguerre_order + 1
            ):
                product = laguerre_product(
                    bessel_order,
                    bessel_laguerre_order,
                    auxiliary_laguerre_order,
                    output_order,
                    bessel_order,
                )
                if float64_final_contraction:
                    total += (
                        float(leading)
                        * float(radial_kernels[output_order])
                        * float(product)
                    )
                else:
                    total += leading * radial_kernels[output_order] * product
    return total


def gyroaveraged_polarization_coefficient(
    spherical_order: int,
    spherical_radial_order: int,
    bessel_order: int,
    bessel_argument: float,
    *,
    maximum_bessel_laguerre_order: int = 24,
    digits: int = 80,
) -> float:
    r"""Return equation (3.41)'s finite-``b`` polarization coefficient."""

    if any(
        order < 0
        for order in (
            spherical_order,
            spherical_radial_order,
            bessel_order,
            maximum_bessel_laguerre_order,
        )
    ):
        raise ValueError("basis and truncation orders must be >= 0")
    if bessel_order > spherical_order:
        raise ValueError("bessel_order must be <= spherical_order")
    if not math.isfinite(bessel_argument) or bessel_argument < 0.0:
        raise ValueError("bessel_argument must be finite and >= 0")
    if digits < 16:
        raise ValueError("digits must be >= 16")

    import mpmath as mp

    with mp.workdps(digits):
        coefficient = _gyroaveraged_polarization_coefficient_mp(
            spherical_order,
            spherical_radial_order,
            bessel_order,
            bessel_argument,
            maximum_bessel_laguerre_order,
            mp,
        )
    return float(coefficient)


def _coulomb_coefficient_functions(
    mp: Any,
    sigma: Any,
    tau: Any,
) -> tuple[Callable[..., Any], ...]:
    """Cache the kperp-independent basis algebra for one species pair."""

    coulomb_e, coulomb_E = _cached_coulomb_integrals_mp(mp.sqrt(tau / sigma), mp)

    @cache
    def associated_laguerre(
        polynomial_order: int,
        tensor_order: Any,
        monomial_order: int,
    ) -> Any:
        return _associated_laguerre_monomial_coefficient_mp(
            polynomial_order,
            tensor_order,
            monomial_order,
            mp,
        )

    @cache
    def legendre_monomial(order: int, power: int) -> Any:
        return _legendre_monomial_coefficient_mp(order, power, mp)

    @cache
    def spherical_polynomial(
        legendre_order: int,
        radial_order: int,
    ) -> tuple[tuple[int, int, Any], ...]:
        return _spherical_polynomial_mp(
            legendre_order,
            radial_order,
            mp,
            associated_laguerre=associated_laguerre,
            legendre_monomial=legendre_monomial,
        )

    @cache
    def hermite_gaussian_moment(hermite_order: int, power: int) -> Any:
        return _hermite_gaussian_moment_mp(hermite_order, power, mp)

    @cache
    def laguerre_exponential_moment(laguerre_order: int, power: int) -> Any:
        return _laguerre_exponential_moment_mp(
            laguerre_order,
            power,
            mp,
            associated_laguerre=associated_laguerre,
        )

    @cache
    def base_transform(
        legendre_order: int,
        radial_order: int,
        hermite_order: int,
        laguerre_order: int,
    ) -> Any:
        return _legendre_to_hermite_laguerre_mp(
            legendre_order,
            radial_order,
            hermite_order,
            laguerre_order,
            mp,
            spherical_polynomial=spherical_polynomial,
            hermite_gaussian_moment=hermite_gaussian_moment,
            laguerre_exponential_moment=laguerre_exponential_moment,
        )

    @cache
    def associated_spherical_polynomial(
        spherical_order: int,
        spherical_radial_order: int,
        bessel_order: int,
    ) -> tuple[tuple[int, int, Any], ...]:
        return _associated_spherical_polynomial_mp(
            spherical_order,
            spherical_radial_order,
            bessel_order,
            mp,
            associated_laguerre=associated_laguerre,
            legendre_monomial=legendre_monomial,
        )

    @cache
    def inverse_associated(
        spherical_order: int,
        spherical_radial_order: int,
        bessel_order: int,
        hermite_order: int,
        laguerre_order: int,
    ) -> Any:
        return _associated_legendre_to_hermite_laguerre_mp(
            spherical_order,
            spherical_radial_order,
            bessel_order,
            hermite_order,
            laguerre_order,
            mp,
            associated_spherical_polynomial=associated_spherical_polynomial,
            hermite_gaussian_moment=hermite_gaussian_moment,
            laguerre_exponential_moment=laguerre_exponential_moment,
        )

    @cache
    def inverse_product(
        associated_order: int,
        associated_polynomial_order: int,
        laguerre_order: int,
        output_order: int,
        radial_power: int,
    ) -> Any:
        return _laguerre_product_expansion_coefficient_mp(
            associated_order,
            associated_polynomial_order,
            laguerre_order,
            output_order,
            radial_power,
            mp,
            monomial_coefficient=associated_laguerre,
            laguerre_moment=laguerre_exponential_moment,
        )

    @cache
    def speed_moment(p: int, j: int, speed_power: int) -> tuple[Any, Any]:
        return _coulomb_speed_moments_mp(
            p,
            j,
            speed_power,
            sigma,
            tau,
            1,
            mp,
            coulomb_e=coulomb_e,
            coulomb_E=coulomb_E,
        )

    return (
        base_transform,
        associated_laguerre,
        legendre_monomial,
        inverse_associated,
        inverse_product,
        speed_moment,
        associated_spherical_polynomial,
        hermite_gaussian_moment,
        laguerre_exponential_moment,
    )


def _precompute_coulomb_speed_coefficients(
    coefficient_functions: tuple[Callable[..., Any], ...],
    *,
    maximum_spherical_order: int,
    maximum_spherical_radial_order: int,
    maximum_speed_power: int,
    worker_count: int,
) -> tuple[Callable[..., Any], ...]:
    """Populate wavelength-independent speed moments before row assembly."""

    if worker_count <= 1:
        return coefficient_functions
    direct_speed = coefficient_functions[5]
    keys = tuple(
        (p, j, speed_power)
        for p in range(maximum_spherical_order + 1)
        for j in range(maximum_spherical_radial_order + 1)
        for speed_power in range(maximum_speed_power + 1)
    )

    def build(index: int) -> tuple[tuple[int, int, int], tuple[Any, Any]]:
        key = keys[index]
        return key, direct_speed(*key)

    values = dict(_forked_ordered_map(build, len(keys), min(worker_count, len(keys))))
    functions = list(coefficient_functions)
    functions[5] = lambda p, j, speed_power: values[(p, j, speed_power)]
    return tuple(functions)


def coulomb_nonpolarized_moment_matrices(
    maximum_hermite_order: int,
    maximum_laguerre_order: int,
    target_bessel_argument: float,
    mass_ratio: float,
    temperature_ratio: float,
    *,
    source_bessel_argument: float | None = None,
    maximum_spherical_order: int | None = None,
    maximum_spherical_radial_order: int | None = None,
    maximum_angular_bessel_order: int | None = None,
    included_angular_orders: tuple[int, ...] | None = None,
    maximum_bessel_laguerre_order: int = 24,
    digits: int = 80,
    float64_final_contraction: bool = False,
    worker_count: int = 1,
    _coefficient_functions: tuple[Callable[..., Any], ...] | None = None,
    _assembly_cache: dict[str, dict[tuple[Any, ...], Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Generate finite-``b`` Coulomb test and field moment matrices.

    This contracts equations (3.48)--(3.49) of Frei et al. (2021), excluding
    the electrostatic polarization terms in equation (3.50).  Rows and columns
    use Hermite-major ``p * Nl + j`` ordering and the paper's Laguerre
    convention. ``target_bessel_argument`` enters the outer gyroaverage and
    the test moments; ``source_bessel_argument`` enters the field-particle
    source moments and defaults to the target value for like species. Both are
    :math:`B=k_\perp v_{\mathrm{th}}/\Omega`. Every finite truncation is
    explicit so convergence can be assessed before a table is promoted.
    ``maximum_angular_bessel_order`` truncates the angular Bessel harmonic
    :math:`m` independently from the radial Bessel--Laguerre expansion.  The
    default retains every harmonic allowed by the spherical basis.
    ``included_angular_orders`` selects a sorted subset for resumable offline
    generation; summing all single-harmonic blocks recovers the full operator.
    ``worker_count`` partitions angular moment-vector blocks and then complete
    output-Hermite rows; one worker is the portable default and multiple
    workers require POSIX ``fork``.
    ``float64_final_contraction`` retains multiprecision coefficient generation
    but performs the final projection-vector products in the archive's float64
    precision. The exact path remains the default verification oracle.
    """

    if maximum_hermite_order < 0:
        raise ValueError("maximum_hermite_order must be >= 0")
    if maximum_laguerre_order < 0:
        raise ValueError("maximum_laguerre_order must be >= 0")
    if not math.isfinite(target_bessel_argument) or target_bessel_argument < 0.0:
        raise ValueError("target_bessel_argument must be finite and >= 0")
    source_bessel_argument = (
        target_bessel_argument
        if source_bessel_argument is None
        else source_bessel_argument
    )
    if not math.isfinite(source_bessel_argument) or source_bessel_argument < 0.0:
        raise ValueError("source_bessel_argument must be finite and >= 0")
    if mass_ratio <= 0.0 or not math.isfinite(mass_ratio):
        raise ValueError("mass_ratio must be finite and > 0")
    if temperature_ratio <= 0.0 or not math.isfinite(temperature_ratio):
        raise ValueError("temperature_ratio must be finite and > 0")
    if maximum_bessel_laguerre_order < 0:
        raise ValueError("maximum_bessel_laguerre_order must be >= 0")
    if maximum_angular_bessel_order is not None and maximum_angular_bessel_order < 0:
        raise ValueError("maximum_angular_bessel_order must be >= 0")
    if digits < 16:
        raise ValueError("digits must be >= 16")
    if worker_count < 1:
        raise ValueError("worker_count must be >= 1")

    maximum_degree = maximum_hermite_order + 2 * maximum_laguerre_order
    spherical_limit = (
        maximum_degree if maximum_spherical_order is None else maximum_spherical_order
    )
    radial_limit = (
        maximum_degree // 2
        if maximum_spherical_radial_order is None
        else maximum_spherical_radial_order
    )
    if spherical_limit < 0:
        raise ValueError("maximum_spherical_order must be >= 0")
    if radial_limit < 0:
        raise ValueError("maximum_spherical_radial_order must be >= 0")
    angular_limit = (
        spherical_limit
        if maximum_angular_bessel_order is None
        else min(spherical_limit, maximum_angular_bessel_order)
    )

    import mpmath as mp

    n_laguerre = maximum_laguerre_order + 1
    n_modes = (maximum_hermite_order + 1) * n_laguerre
    with mp.workdps(digits):
        target_b = mp.mpf(target_bessel_argument)
        source_b = mp.mpf(source_bessel_argument)
        sigma = mp.mpf(mass_ratio)
        tau = mp.mpf(temperature_ratio)
        half_b = target_b / 2
        quarter_b_squared = half_b * half_b
        drift_kinetic = target_b == 0
        active_angular_orders = _resolve_collision_angular_orders(
            angular_limit,
            included_angular_orders,
            drift_kinetic=drift_kinetic,
        )
        bessel_orders = (
            (0,) if drift_kinetic else tuple(range(maximum_bessel_laguerre_order + 1))
        )
        assembly_cache = {} if _assembly_cache is None else _assembly_cache
        moment_cache = assembly_cache.setdefault("spherical_moment", {})
        bessel_kernel_cache = assembly_cache.setdefault("spherical_bessel_kernel", {})
        bessel_projection_cache = assembly_cache.setdefault(
            "spherical_bessel_projection", {}
        )
        speed_cache = assembly_cache.setdefault("integrated_speed", {})
        product_cache = assembly_cache.setdefault("laguerre_product", {})
        inverse_cache = assembly_cache.setdefault("inverse_transform", {})

        coefficient_functions = (
            _coulomb_coefficient_functions(mp, sigma, tau)
            if _coefficient_functions is None
            else _coefficient_functions
        )
        (
            _base_transform,
            associated_laguerre,
            _legendre_monomial,
            inverse_associated,
            inverse_product,
            speed_moment,
            associated_spherical_polynomial,
            hermite_gaussian_moment,
            laguerre_exponential_moment,
        ) = coefficient_functions

        def spherical_moment(
            p: int,
            j: int,
            m: int,
            g: int,
            s: int,
            *,
            source: bool,
        ) -> Any:
            wavelength = source_b if source else target_b
            key = (float(wavelength), p, j, m, g, s)
            if key not in moment_cache:
                kernel_key = (
                    float(wavelength),
                    m,
                    maximum_bessel_laguerre_order,
                )
                if kernel_key not in bessel_kernel_cache:
                    bessel_kernel_cache[kernel_key] = _bessel_laguerre_kernels_mp(
                        wavelength,
                        m,
                        maximum_bessel_laguerre_order + 1,
                        mp,
                    )

                def projected_product(
                    auxiliary_laguerre_order: int,
                    input_laguerre_order: int,
                ) -> float:
                    projection_key = (
                        float(wavelength),
                        m,
                        auxiliary_laguerre_order,
                        input_laguerre_order,
                    )
                    if projection_key not in bessel_projection_cache:
                        bessel_projection_cache[projection_key] = sum(
                            float(
                                inverse_product(
                                    m,
                                    bessel_laguerre_order,
                                    auxiliary_laguerre_order,
                                    input_laguerre_order,
                                    m,
                                )
                            )
                            * float(
                                bessel_kernel_cache[kernel_key][bessel_laguerre_order]
                            )
                            for bessel_laguerre_order in range(
                                maximum_bessel_laguerre_order + 1
                            )
                        )
                    return float(bessel_projection_cache[projection_key])

                moment_cache[key] = _gyroaveraged_spherical_moment_coefficient_mp(
                    p,
                    j,
                    m,
                    g,
                    s,
                    source_b if source else target_b,
                    maximum_bessel_laguerre_order,
                    mp,
                    associated_transform=inverse_associated,
                    laguerre_product=inverse_product,
                    bessel_kernels=bessel_kernel_cache[kernel_key],
                    bessel_product_projection=(
                        projected_product if float64_final_contraction else None
                    ),
                    float64_final_contraction=float64_final_contraction,
                )
            return moment_cache[key]

        def integrated_speed(p: int, j: int, t: int) -> tuple[Any, Any]:
            key = (p, j, t)
            if key not in speed_cache:
                test_speed = mp.mpf(0)
                field_speed = mp.mpf(0)
                for speed_power in range(t + 1):
                    laguerre_coefficient = associated_laguerre(
                        t,
                        p,
                        speed_power,
                    )
                    test_term, field_term = speed_moment(p, j, speed_power)
                    test_speed += laguerre_coefficient * test_term
                    field_speed += laguerre_coefficient * field_term
                speed_cache[key] = (test_speed, field_speed)
            return speed_cache[key]

        def laguerre_product(
            m: int,
            n: int,
            output_laguerre: int,
            product_order: int,
        ) -> Any:
            key = (m, n, output_laguerre, product_order)
            if key not in product_cache:
                product_cache[key] = inverse_product(
                    m,
                    n,
                    output_laguerre,
                    product_order,
                    0,
                )
            return product_cache[key]

        def inverse_transform(
            output_hermite: int,
            product_order: int,
            p: int,
            speed_order: int,
            m: int,
        ) -> Any:
            key = (output_hermite, product_order, p, speed_order, m)
            if key not in inverse_cache:
                inverse_cache[key] = _hermite_laguerre_to_associated_legendre_mp(
                    output_hermite,
                    product_order,
                    p,
                    speed_order,
                    m,
                    mp,
                    associated_spherical_polynomial=associated_spherical_polynomial,
                    hermite_gaussian_moment=hermite_gaussian_moment,
                    laguerre_exponential_moment=laguerre_exponential_moment,
                )
            return inverse_cache[key]

        moment_orders = tuple(
            (p, j, m)
            for p in range(spherical_limit + 1)
            for j in range(radial_limit + 1)
            for m in active_angular_orders
            if m <= p
        )

        def build_moment_vector(p: int, j: int, m: int) -> tuple[Any, Any]:
            return (
                tuple(
                    spherical_moment(
                        p,
                        j,
                        m,
                        input_hermite,
                        input_laguerre,
                        source=False,
                    )
                    for input_hermite in range(maximum_hermite_order + 1)
                    for input_laguerre in range(n_laguerre)
                ),
                tuple(
                    spherical_moment(
                        p,
                        j,
                        m,
                        input_hermite,
                        input_laguerre,
                        source=True,
                    )
                    for input_hermite in range(maximum_hermite_order + 1)
                    for input_laguerre in range(n_laguerre)
                ),
            )

        if float64_final_contraction and worker_count > 1 and not drift_kinetic:
            if len(active_angular_orders) == 1:
                angular_order = active_angular_orders[0]
                for wavelength in {target_b, source_b}:
                    kernel_key = (
                        float(wavelength),
                        angular_order,
                        maximum_bessel_laguerre_order,
                    )
                    bessel_kernel_cache.setdefault(
                        kernel_key,
                        _bessel_laguerre_kernels_mp(
                            wavelength,
                            angular_order,
                            maximum_bessel_laguerre_order + 1,
                            mp,
                        ),
                    )
                moment_worker_count = min(worker_count, len(moment_orders))
                moment_tasks = tuple(
                    tuple(moment_orders[index::moment_worker_count])
                    for index in range(moment_worker_count)
                )
            else:
                moment_worker_count = len(active_angular_orders)
                moment_tasks = tuple(
                    tuple(
                        orders for orders in moment_orders if orders[2] == angular_order
                    )
                    for angular_order in active_angular_orders
                )

            def build_moment_vectors(
                task_index: int,
            ) -> tuple[int, tuple[tuple[tuple[int, int, int], tuple[Any, Any]], ...]]:
                return task_index, tuple(
                    ((p, j, m), build_moment_vector(p, j, m))
                    for p, j, m in moment_tasks[task_index]
                )

            moment_blocks = _forked_ordered_map(
                build_moment_vectors,
                len(moment_tasks),
                moment_worker_count,
            )
            moment_vectors = {
                key: values
                for _task_index, block in moment_blocks
                for key, values in block
            }
        else:
            moment_vectors = {
                (p, j, m): build_moment_vector(p, j, m) for p, j, m in moment_orders
            }
        angular_weights = {}
        for p, j, m in moment_orders:
            sigma_pj = (
                mp.factorial(p)
                * mp.gamma(p + j + mp.mpf("1.5"))
                / (mp.power(2, p) * mp.gamma(p + mp.mpf("1.5")) * mp.factorial(j))
            )
            angular_weights[(p, j, m)] = (
                (1 if m == 0 else 2)
                * mp.power(2, p)
                * mp.factorial(p) ** 2
                / (sigma_pj * mp.factorial(2 * p) * (2 * p + 1))
            )
        exponential = mp.exp(-quarter_b_squared)
        bessel_factors = {
            m: tuple(
                exponential * quarter_b_squared**n * half_b**m / mp.factorial(n + m)
                for n in bessel_orders
            )
            for m in active_angular_orders
        }
        weighted_products = {
            (m, output_laguerre): _bessel_weighted_laguerre_products_mp(
                m,
                output_laguerre,
                bessel_orders,
                bessel_factors[m],
                laguerre_product,
                mp,
            )
            for m in active_angular_orders
            for output_laguerre in range(n_laguerre)
        }
        grouped_moments: dict[tuple[int, int], list[tuple[Any, ...]]] = {}
        for p, j, m in moment_orders:
            test_vector, field_vector = moment_vectors[(p, j, m)]
            if not any(
                value != 0 for vector in (test_vector, field_vector) for value in vector
            ):
                continue
            grouped_moments.setdefault((p, m), []).append(
                (
                    j,
                    (
                        np.asarray(test_vector, dtype=np.float64)
                        if float64_final_contraction
                        else test_vector
                    ),
                    (
                        np.asarray(field_vector, dtype=np.float64)
                        if float64_final_contraction
                        else field_vector
                    ),
                    angular_weights[(p, j, m)],
                )
            )
        active_moment_groups = tuple(
            (p, m, tuple(entries)) for (p, m), entries in grouped_moments.items()
        )

        def build_hermite_rows(
            output_hermite: int,
        ) -> tuple[int, np.ndarray, np.ndarray]:
            test_rows = (
                np.zeros((n_laguerre, n_modes), dtype=np.float64)
                if float64_final_contraction
                else mp.matrix(n_laguerre, n_modes)
            )
            field_rows = (
                np.zeros((n_laguerre, n_modes), dtype=np.float64)
                if float64_final_contraction
                else mp.matrix(n_laguerre, n_modes)
            )
            output_normalization = mp.sqrt(
                mp.power(2, output_hermite) * mp.factorial(output_hermite)
            )
            for output_laguerre in range(n_laguerre):
                for p, m, radial_moments in active_moment_groups:
                    speed_weights: dict[int, Any] = {}
                    for product_order, weighted_product in enumerate(
                        weighted_products[(m, output_laguerre)]
                    ):
                        if weighted_product == 0:
                            continue
                        if p > output_hermite + m + 2 * product_order:
                            continue
                        maximum_speed_order = product_order + (output_hermite + m) // 2
                        for speed_order in range(maximum_speed_order + 1):
                            inverse = inverse_transform(
                                output_hermite,
                                product_order,
                                p,
                                speed_order,
                                m,
                            )
                            if inverse == 0:
                                continue
                            common = weighted_product * inverse / output_normalization
                            speed_weights[speed_order] = (
                                speed_weights.get(
                                    speed_order,
                                    mp.mpf(0),
                                )
                                + common
                            )
                    for (
                        j,
                        test_moment_vector,
                        field_moment_vector,
                        angular,
                    ) in radial_moments:
                        test_output_coefficient = mp.mpf(0)
                        field_output_coefficient = mp.mpf(0)
                        for speed_order, weight in speed_weights.items():
                            test_speed, field_speed = integrated_speed(
                                p,
                                j,
                                speed_order,
                            )
                            test_output_coefficient += weight * test_speed
                            field_output_coefficient += weight * field_speed
                        test_output_coefficient *= angular
                        field_output_coefficient *= angular
                        if test_output_coefficient != 0:
                            if float64_final_contraction:
                                test_rows[output_laguerre] += float(
                                    test_output_coefficient
                                ) * np.asarray(test_moment_vector)
                            else:
                                for column, moment in enumerate(test_moment_vector):
                                    test_rows[output_laguerre, column] += (
                                        test_output_coefficient * moment
                                    )
                        if field_output_coefficient != 0:
                            if float64_final_contraction:
                                field_rows[output_laguerre] += float(
                                    field_output_coefficient
                                ) * np.asarray(field_moment_vector)
                            else:
                                for column, moment in enumerate(field_moment_vector):
                                    field_rows[output_laguerre, column] += (
                                        field_output_coefficient * moment
                                    )
            return (
                output_hermite,
                (
                    test_rows
                    if float64_final_contraction
                    else np.asarray(test_rows.tolist(), dtype=np.float64)
                ),
                (
                    field_rows
                    if float64_final_contraction
                    else np.asarray(field_rows.tolist(), dtype=np.float64)
                ),
            )

        row_blocks = _forked_ordered_map(
            build_hermite_rows,
            maximum_hermite_order + 1,
            min(worker_count, maximum_hermite_order + 1),
        )
        test = np.empty((n_modes, n_modes))
        field = np.empty((n_modes, n_modes))
        for output_hermite, test_rows, field_rows in row_blocks:
            row_slice = slice(
                output_hermite * n_laguerre,
                (output_hermite + 1) * n_laguerre,
            )
            test[row_slice] = test_rows
            field[row_slice] = field_rows
    return test, field


def coulomb_drift_kinetic_moment_matrices(
    maximum_hermite_order: int,
    maximum_laguerre_order: int,
    mass_ratio: float,
    temperature_ratio: float,
    *,
    maximum_spherical_order: int | None = None,
    maximum_spherical_radial_order: int | None = None,
    digits: int = 80,
    float64_final_contraction: bool = False,
    worker_count: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Generate drift-kinetic Coulomb test and field moment matrices.

    This evaluates equations (3.53)--(3.56) of Frei et al. (2021) directly.
    At zero Larmor radius only the azimuthal harmonic ``m=0`` remains, so the
    Bessel and Laguerre-product sums used by the finite-wavelength generator
    collapse exactly. Rows and columns use Hermite-major ``p * Nl + j``
    ordering and the paper's Laguerre convention.

    The spherical and radial limits remain explicit because collision-table
    promotion requires a resolved hierarchy, not only a resolved runtime
    Hermite--Laguerre state. ``worker_count`` partitions independent Coulomb
    speed moments through POSIX ``fork`` while keeping one worker as the
    portable default. ``float64_final_contraction`` converts only the final
    dense product after all coefficients have been evaluated at ``digits``
    precision; it is intended for bounded high-order artifact generation.
    """

    if maximum_hermite_order < 0:
        raise ValueError("maximum_hermite_order must be >= 0")
    if maximum_laguerre_order < 0:
        raise ValueError("maximum_laguerre_order must be >= 0")
    if mass_ratio <= 0.0 or not math.isfinite(mass_ratio):
        raise ValueError("mass_ratio must be finite and > 0")
    if temperature_ratio <= 0.0 or not math.isfinite(temperature_ratio):
        raise ValueError("temperature_ratio must be finite and > 0")
    if digits < 16:
        raise ValueError("digits must be >= 16")
    if worker_count < 1:
        raise ValueError("worker_count must be >= 1")

    maximum_degree = maximum_hermite_order + 2 * maximum_laguerre_order
    spherical_limit = (
        maximum_degree if maximum_spherical_order is None else maximum_spherical_order
    )
    radial_limit = (
        maximum_degree // 2
        if maximum_spherical_radial_order is None
        else maximum_spherical_radial_order
    )
    if spherical_limit < 0:
        raise ValueError("maximum_spherical_order must be >= 0")
    if radial_limit < 0:
        raise ValueError("maximum_spherical_radial_order must be >= 0")

    import mpmath as mp

    n_laguerre = maximum_laguerre_order + 1
    n_modes = (maximum_hermite_order + 1) * n_laguerre
    moment_orders = tuple(
        (p, j) for p in range(spherical_limit + 1) for j in range(radial_limit + 1)
    )

    with mp.workdps(digits):
        sigma = mp.mpf(mass_ratio)
        tau = mp.mpf(temperature_ratio)
        chi = mp.sqrt(tau / sigma)

        coulomb_e, coulomb_E = _cached_coulomb_integrals_mp(chi, mp)

        @cache
        def spherical_polynomial(p: int, j: int) -> tuple[tuple[int, int, Any], ...]:
            return _spherical_polynomial_mp(p, j, mp)

        @cache
        def hermite_gaussian_moment(g: int, power: int) -> Any:
            return _hermite_gaussian_moment_mp(g, power, mp)

        @cache
        def laguerre_exponential_moment(h: int, power: int) -> Any:
            return _laguerre_exponential_moment_mp(h, power, mp)

        @cache
        def transform(p: int, j: int, g: int, h: int) -> Any:
            left_degree = p + 2 * j
            right_degree = g + 2 * h
            if right_degree > left_degree or (left_degree - right_degree) % 2:
                return mp.mpf(0)
            return _legendre_to_hermite_laguerre_mp(
                p,
                j,
                g,
                h,
                mp,
                spherical_polynomial=spherical_polynomial,
                hermite_gaussian_moment=hermite_gaussian_moment,
                laguerre_exponential_moment=laguerre_exponential_moment,
            )

        @cache
        def direct_speed_moment(p: int, j: int, speed_power: int) -> tuple[Any, Any]:
            return _coulomb_speed_moments_mp(
                p,
                j,
                speed_power,
                sigma,
                tau,
                1,
                mp,
                coulomb_e=coulomb_e,
                coulomb_E=coulomb_E,
            )

        speed_keys: set[tuple[int, int, int]] = set()
        for output_hermite in range(maximum_hermite_order + 1):
            for output_laguerre in range(n_laguerre):
                output_degree = output_hermite + 2 * output_laguerre
                maximum_speed_order = output_laguerre + output_hermite // 2
                for p, j in moment_orders:
                    if p > output_degree:
                        continue
                    for speed_order in range(maximum_speed_order + 1):
                        if output_laguerre > speed_order + p // 2:
                            continue
                        if output_degree > p + 2 * speed_order:
                            continue
                        if (p + 2 * speed_order - output_degree) % 2:
                            continue
                        speed_keys.update(
                            (p, j, speed_power)
                            for speed_power in range(speed_order + 1)
                        )
        ordered_speed_keys = tuple(sorted(speed_keys))

        def build_speed_moment(index: int):
            key = ordered_speed_keys[index]
            return key, direct_speed_moment(*key)

        speed_moments = dict(
            _forked_ordered_map(
                build_speed_moment,
                len(ordered_speed_keys),
                min(worker_count, len(ordered_speed_keys)),
            )
        )

        @cache
        def integrated_speed(p: int, j: int, t: int) -> tuple[Any, Any]:
            test_speed = mp.mpf(0)
            field_speed = mp.mpf(0)
            for speed_power in range(t + 1):
                laguerre_coefficient = _associated_laguerre_monomial_coefficient_mp(
                    t, p, speed_power, mp
                )
                test_term, field_term = speed_moments[(p, j, speed_power)]
                test_speed += laguerre_coefficient * test_term
                field_speed += laguerre_coefficient * field_term
            return test_speed, field_speed

        # Equation (3.54): map runtime Hermite--Laguerre coefficients to the
        # particle spherical moments used by both collision components.
        moment_map = mp.matrix(len(moment_orders), n_modes)
        for moment_index, (p, j) in enumerate(moment_orders):
            for g in range(maximum_hermite_order + 1):
                normalization = mp.sqrt(mp.power(2, g) * mp.factorial(g))
                for h in range(n_laguerre):
                    moment_map[moment_index, g * n_laguerre + h] = (
                        transform(p, j, g, h) * normalization
                    )

        test_projection = mp.matrix(n_modes, len(moment_orders))
        field_projection = mp.matrix(n_modes, len(moment_orders))
        for output_hermite in range(maximum_hermite_order + 1):
            output_normalization = mp.sqrt(
                mp.power(2, output_hermite) * mp.factorial(output_hermite)
            )
            for output_laguerre in range(n_laguerre):
                row = output_hermite * n_laguerre + output_laguerre
                maximum_speed_order = output_laguerre + output_hermite // 2
                for moment_index, (p, j) in enumerate(moment_orders):
                    if p > output_hermite + 2 * output_laguerre:
                        continue
                    sigma_pj = (
                        mp.factorial(p)
                        * mp.gamma(p + j + mp.mpf("1.5"))
                        / (
                            mp.power(2, p)
                            * mp.gamma(p + mp.mpf("1.5"))
                            * mp.factorial(j)
                        )
                    )
                    angular = (
                        mp.power(2, p)
                        * mp.factorial(p) ** 2
                        / (sigma_pj * mp.factorial(2 * p) * (2 * p + 1))
                    )
                    test_coefficient = mp.mpf(0)
                    field_coefficient = mp.mpf(0)
                    for speed_order in range(maximum_speed_order + 1):
                        if output_laguerre > speed_order + p // 2:
                            continue
                        inverse_prefactor = (
                            mp.sqrt(mp.pi)
                            * mp.power(2, output_hermite)
                            * mp.factorial(output_hermite)
                            * mp.factorial(speed_order)
                            * (p + mp.mpf("0.5"))
                            / mp.gamma(speed_order + p + mp.mpf("1.5"))
                        )
                        inverse = inverse_prefactor * transform(
                            p,
                            speed_order,
                            output_hermite,
                            output_laguerre,
                        )
                        if inverse == 0:
                            continue
                        test_speed, field_speed = integrated_speed(p, j, speed_order)
                        common = angular * inverse / output_normalization
                        test_coefficient += common * test_speed
                        field_coefficient += common * field_speed
                    test_projection[row, moment_index] = test_coefficient
                    field_projection[row, moment_index] = field_coefficient

        if float64_final_contraction:
            moment_map_float = np.asarray(moment_map.tolist(), dtype=np.float64)
            test = (
                np.asarray(test_projection.tolist(), dtype=np.float64)
                @ moment_map_float
            )
            field = (
                np.asarray(field_projection.tolist(), dtype=np.float64)
                @ moment_map_float
            )
        else:
            test = np.asarray((test_projection * moment_map).tolist(), dtype=np.float64)
            field = np.asarray(
                (field_projection * moment_map).tolist(), dtype=np.float64
            )
    return test, field


def original_sugama_like_species_moment_matrices(
    coulomb_test_matrix: np.ndarray,
    maximum_hermite_order: int,
    maximum_laguerre_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Build the equal-temperature like-species original-Sugama matrices.

    At equal temperature the original-Sugama test component is the Coulomb
    test component. Its field component is the self-adjoint, rank-one
    restoration of parallel momentum and thermal energy described by Sugama,
    Watanabe & Nunami (2009). In an orthonormal Hermite--Laguerre basis each
    invariant ``u`` contributes ``-(T u)(T u)^T / (u^T T u)``. The input and
    output use the paper's Laguerre convention, matching the offline Coulomb
    generator.
    """

    if maximum_hermite_order < 2:
        raise ValueError("maximum_hermite_order must be >= 2")
    if maximum_laguerre_order < 1:
        raise ValueError("maximum_laguerre_order must be >= 1")
    n_laguerre = maximum_laguerre_order + 1
    mode_count = (maximum_hermite_order + 1) * n_laguerre
    test_paper = np.asarray(coulomb_test_matrix, dtype=float)
    if test_paper.shape != (mode_count, mode_count):
        raise ValueError("coulomb_test_matrix shape does not match the resolution")
    if not np.all(np.isfinite(test_paper)):
        raise ValueError("coulomb_test_matrix must contain only finite values")

    convention_sign = np.asarray(
        [
            (-1.0) ** laguerre
            for _hermite in range(maximum_hermite_order + 1)
            for laguerre in range(n_laguerre)
        ]
    )
    convention = convention_sign[:, None] * convention_sign[None, :]
    test = convention * test_paper
    momentum = np.zeros(mode_count)
    momentum[n_laguerre] = 1.0
    energy = np.zeros(mode_count)
    energy[1] = 1.0
    energy[2 * n_laguerre] = 1 / np.sqrt(2.0)
    field = np.zeros_like(test)
    for invariant in (momentum, energy):
        image = test @ invariant
        denominator = float(invariant @ image)
        if denominator >= 0.0 or not math.isfinite(denominator):
            raise ValueError(
                "coulomb_test_matrix must dissipate momentum and thermal energy"
            )
        field -= np.outer(image, image) / denominator
    return test_paper.copy(), convention * field


def finite_wavelength_original_sugama_like_species_moment_matrices(
    coulomb_test_table: np.ndarray,
    bessel_arguments: np.ndarray,
    maximum_hermite_order: int,
    maximum_laguerre_order: int,
    *,
    quadrature_order: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Build equal-species finite-wavelength original-Sugama tables.

    For like species, Frei et al. (2021), equations (3.72)--(3.76), reduce the
    original-Sugama test component to the Coulomb test component. Equations
    (3.79) and (3.65), (3.68)--(3.69), and (3.80) make the field
    component the sum of explicit parallel-flow, perpendicular-flow, and
    thermal-energy rank-one terms. Their velocity projections are evaluated
    with product Gauss--Hermite/Laguerre quadrature. Inputs and outputs use the
    runtime's signed-Laguerre, Hermite-major convention.
    """

    if maximum_hermite_order < 2:
        raise ValueError("maximum_hermite_order must be >= 2")
    if maximum_laguerre_order < 1:
        raise ValueError("maximum_laguerre_order must be >= 1")
    if quadrature_order < 32:
        raise ValueError("quadrature_order must be >= 32")
    grid = np.asarray(bessel_arguments, dtype=float)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("bessel_arguments must be one-dimensional and nonempty")
    if np.any(~np.isfinite(grid)) or np.any(grid < 0.0) or np.any(np.diff(grid) <= 0):
        raise ValueError(
            "bessel_arguments must be finite, nonnegative, and strictly increasing"
        )
    n_laguerre = maximum_laguerre_order + 1
    mode_count = (maximum_hermite_order + 1) * n_laguerre
    test = np.asarray(coulomb_test_table, dtype=float)
    if test.shape != (grid.size, mode_count, mode_count):
        raise ValueError("coulomb_test_table shape does not match grid and resolution")
    if not np.all(np.isfinite(test)):
        raise ValueError("coulomb_test_table must contain only finite values")

    from scipy.special import (
        erf,
        eval_hermite,
        eval_laguerre,
        gammaln,
        jv,
        roots_hermite,
        roots_laguerre,
    )

    laguerre_sign = np.asarray(
        [
            (-1.0) ** laguerre
            for _hermite in range(maximum_hermite_order + 1)
            for laguerre in range(n_laguerre)
        ]
    )
    convention = laguerre_sign[:, None] * laguerre_sign[None, :]
    hermite_normalization = np.exp(
        0.5
        * (
            np.arange(maximum_hermite_order + 1) * np.log(2.0)
            + gammaln(np.arange(maximum_hermite_order + 1) + 1)
        )
    )

    def field_at_wavelength(bessel_argument: float, node_count: int) -> np.ndarray:
        hermite_nodes, hermite_weights = roots_hermite(node_count)
        laguerre_nodes, laguerre_weights = roots_laguerre(node_count)
        parallel_speed = hermite_nodes[:, None]
        perpendicular_energy = laguerre_nodes[None, :]
        speed = np.sqrt(parallel_speed**2 + perpendicular_energy)
        erf_prime = 2.0 / np.sqrt(np.pi) * np.exp(-(speed**2))
        chandrasekhar = (erf(speed) - speed * erf_prime) / (2.0 * speed**2)
        bessel_argument_grid = bessel_argument * np.sqrt(perpendicular_energy)
        common_flow = -8.0 * np.sqrt(2.0) * chandrasekhar / speed
        responses = (
            common_flow * parallel_speed * jv(0, bessel_argument_grid),
            common_flow * np.sqrt(perpendicular_energy) * jv(1, bessel_argument_grid),
            -4.0
            * (erf(speed) - 2.0 * speed * erf_prime)
            / speed
            * jv(0, bessel_argument_grid),
        )
        weights = hermite_weights[:, None] * laguerre_weights[None, :] / np.sqrt(np.pi)
        hermite = np.asarray(
            [
                eval_hermite(order, hermite_nodes) / hermite_normalization[order]
                for order in range(maximum_hermite_order + 1)
            ]
        )
        laguerre = np.asarray(
            [eval_laguerre(order, laguerre_nodes) for order in range(n_laguerre)]
        )
        vectors = tuple(
            laguerre_sign
            * np.einsum(
                "ph,jx,hx,hx->pj",
                hermite,
                laguerre,
                weights,
                response,
                optimize=True,
            ).reshape(-1)
            for response in responses
        )
        # The generated collision tables use twice the paper's nu_ab. These
        # denominators carry the same factor and recover the verified b=0
        # matrix to roundoff.
        gamma = -8.0 / (3.0 * np.sqrt(2.0 * np.pi))
        eta = -2.0 * np.sqrt(2.0 / np.pi)
        return (
            -np.outer(vectors[0], vectors[0]) / gamma
            - np.outer(vectors[1], vectors[1]) / gamma
            - np.outer(vectors[2], vectors[2]) / eta
        )

    field = np.empty_like(test)
    for wavelength_index, bessel_argument in enumerate(grid):
        if bessel_argument == 0.0:
            # At b=0 the finite-wavelength equations reduce exactly to the
            # drift-kinetic field block. Avoid backend-dependent quadrature
            # leakage into analytically vanishing density channels.
            _paper_test, paper_field = original_sugama_like_species_moment_matrices(
                convention * test[wavelength_index],
                maximum_hermite_order,
                maximum_laguerre_order,
            )
            field[wavelength_index] = convention * paper_field
            continue
        lower = field_at_wavelength(float(bessel_argument), quadrature_order)
        accepted = field_at_wavelength(float(bessel_argument), quadrature_order + 16)
        difference = float(np.linalg.norm(accepted - lower))
        scale = max(float(np.linalg.norm(accepted)), np.finfo(float).tiny)
        if difference > 1.0e-12 and difference / scale > 1.0e-11:
            raise RuntimeError("original-Sugama velocity quadrature did not converge")
        field[wavelength_index] = accepted
    return test.copy(), field


def _improved_sugama_equal_temperature_delta_n_mp(
    correction_order: int,
    mp: Any,
) -> Any:
    """Return the equal-species Braginskii field correction in equation (45)."""

    root_pi = mp.sqrt(mp.pi)
    collision_time = 3 * root_pi / 4
    coulomb_e, coulomb_E = _cached_coulomb_integrals_mp(mp.mpf(1), mp)

    @cache
    def field_speed_moment(k: int, speed_power: int) -> Any:
        return _coulomb_speed_moments_mp(
            1,
            k,
            speed_power,
            mp.mpf(1),
            mp.mpf(1),
            mp.mpf(1),
            mp,
            coulomb_e=coulomb_e,
            coulomb_E=coulomb_E,
        )[1]

    @cache
    def laguerre(order: int, monomial: int) -> Any:
        return _associated_laguerre_monomial_coefficient_mp(order, 1, monomial, mp)

    coulomb_n = mp.matrix(correction_order + 1, correction_order + 1)
    for ell in range(correction_order + 1):
        for k in range(correction_order + 1):
            coulomb_n[ell, k] = sum(
                mp.mpf(2)
                / 3
                * collision_time
                * laguerre(ell, power)
                * field_speed_moment(k, power)
                for power in range(ell + 1)
            )
    delta_n = mp.matrix(correction_order + 1, correction_order + 1)
    for ell in range(correction_order + 1):
        for k in range(correction_order + 1):
            delta_n[ell, k] = (
                coulomb_n[ell, k]
                - coulomb_n[ell, 0] * coulomb_n[0, k] / coulomb_n[0, 0]
            )
    for order in range(correction_order + 1):
        delta_n[0, order] = mp.mpf(0)
        delta_n[order, 0] = mp.mpf(0)
    return delta_n


def improved_sugama_equal_temperature_moment_matrices(
    coulomb_test_matrix: np.ndarray,
    maximum_hermite_order: int,
    maximum_laguerre_order: int,
    *,
    correction_order: int = 3,
    digits: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Build the equal-species improved-Sugama matrices through order ``K``.

    The original-Sugama matrices are augmented by the drift-kinetic field
    correction in Frei, Ernst & Ricci (2022), equation (79).  At equal mass
    and temperature the test correction vanishes.  The field correction uses
    the exact Coulomb and original-Sugama Braginskii ``N`` matrices from their
    equations (45), (60), and (80), followed by the basis transform in
    equations (79) and (81).  Rows and columns retain the paper's Laguerre
    convention.

    ``correction_order`` must fit completely inside the retained basis:
    ``P >= 2 K + 1`` and ``J >= K``.  This prevents a nominal high-order
    correction from silently using an incomplete total-degree shell.
    """

    if correction_order < 0:
        raise ValueError("correction_order must be >= 0")
    if maximum_hermite_order < 2 * correction_order + 1:
        raise ValueError("maximum_hermite_order must be >= 2 * correction_order + 1")
    if maximum_laguerre_order < correction_order:
        raise ValueError("maximum_laguerre_order must be >= correction_order")
    if digits < 16:
        raise ValueError("digits must be >= 16")

    original_test, original_field = original_sugama_like_species_moment_matrices(
        coulomb_test_matrix,
        maximum_hermite_order,
        maximum_laguerre_order,
    )
    import mpmath as mp

    n_laguerre = maximum_laguerre_order + 1
    mode_count = (maximum_hermite_order + 1) * n_laguerre
    with mp.workdps(digits):
        root_pi = mp.sqrt(mp.pi)
        collision_time = 3 * root_pi / 4
        delta_n = _improved_sugama_equal_temperature_delta_n_mp(correction_order, mp)

        def flow_weight(order: int) -> Any:
            double_factorial = mp.fac2(2 * order + 3)
            return 3 * mp.power(2, order) * mp.factorial(order) / double_factorial

        correction = mp.matrix(mode_count, mode_count)
        for ell in range(correction_order + 1):
            output = mp.matrix(mode_count, 1)
            shell_degree = 1 + 2 * ell
            for p in range(maximum_hermite_order + 1):
                remainder = shell_degree - p
                if remainder < 0 or remainder % 2:
                    continue
                j = remainder // 2
                if j > maximum_laguerre_order:
                    continue
                forward = _legendre_to_hermite_laguerre_mp(1, ell, p, j, mp)
                inverse_factor = (
                    root_pi
                    * mp.power(2, p)
                    * mp.factorial(p)
                    * mp.mpf("1.5")
                    * mp.factorial(ell)
                    / mp.gamma(ell + mp.mpf("2.5"))
                )
                inverse = inverse_factor * forward
                output[p * n_laguerre + j] = (
                    inverse
                    / mp.sqrt(mp.power(2, p) * mp.factorial(p))
                    * mp.gamma(ell + mp.mpf("2.5"))
                    / mp.factorial(ell)
                )

            for k in range(correction_order + 1):
                source = mp.matrix(mode_count, 1)
                source_degree = 1 + 2 * k
                for g in range(maximum_hermite_order + 1):
                    remainder = source_degree - g
                    if remainder < 0 or remainder % 2:
                        continue
                    h = remainder // 2
                    if h > maximum_laguerre_order:
                        continue
                    source[g * n_laguerre + h] = _legendre_to_hermite_laguerre_mp(
                        1, k, g, h, mp
                    ) * mp.sqrt(mp.power(2, g) * mp.factorial(g))
                prefactor = (
                    4
                    * flow_weight(ell)
                    * flow_weight(k)
                    / (3 * root_pi * collision_time)
                    * delta_n[ell, k]
                )
                correction += prefactor * output * source.T

    return original_test, original_field + np.asarray(
        correction.tolist(), dtype=np.float64
    )


def finite_wavelength_improved_sugama_like_species_moment_matrices(
    coulomb_test_table: np.ndarray,
    bessel_arguments: np.ndarray,
    maximum_hermite_order: int,
    maximum_laguerre_order: int,
    *,
    correction_order: int = 3,
    quadrature_order: int = 80,
    digits: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Build equal-species finite-wavelength improved-Sugama tables.

    The original-Sugama table is augmented by the equal-species field
    correction in Frei, Ernst & Ricci (2022), equations (61)--(69). The
    velocity projections are evaluated independently with product
    Gauss--Hermite/Laguerre quadrature. A second calculation with 16 more
    nodes must agree before the higher-order result is accepted.
    """

    if correction_order < 0:
        raise ValueError("correction_order must be >= 0")
    if maximum_hermite_order < 2 * correction_order + 1:
        raise ValueError("maximum_hermite_order must be >= 2 * correction_order + 1")
    if maximum_laguerre_order < correction_order:
        raise ValueError("maximum_laguerre_order must be >= correction_order")
    if quadrature_order < 32:
        raise ValueError("quadrature_order must be >= 32")
    if digits < 16:
        raise ValueError("digits must be >= 16")
    original_test, original_field = (
        finite_wavelength_original_sugama_like_species_moment_matrices(
            coulomb_test_table,
            bessel_arguments,
            maximum_hermite_order,
            maximum_laguerre_order,
            quadrature_order=quadrature_order,
        )
    )
    grid = np.asarray(bessel_arguments, dtype=float)
    import mpmath as mp
    from scipy.special import (
        eval_genlaguerre,
        eval_hermite,
        eval_laguerre,
        gammaln,
        jv,
        roots_hermite,
        roots_laguerre,
    )

    with mp.workdps(digits):
        delta_n = np.asarray(
            _improved_sugama_equal_temperature_delta_n_mp(
                correction_order, mp
            ).tolist(),
            dtype=float,
        )
    flow_weight = np.asarray(
        [
            3.0
            * 2.0**order
            * math.factorial(order)
            / math.prod(range(2 * order + 3, 0, -2))
            for order in range(correction_order + 1)
        ]
    )
    collision_time = 3.0 * np.sqrt(np.pi) / 4.0
    correction_kernel = (
        2.0 / collision_time * flow_weight[:, None] * delta_n * flow_weight[None, :]
    )
    n_laguerre = maximum_laguerre_order + 1
    laguerre_sign = np.asarray(
        [
            (-1.0) ** laguerre
            for _hermite in range(maximum_hermite_order + 1)
            for laguerre in range(n_laguerre)
        ]
    )
    convention = laguerre_sign[:, None] * laguerre_sign[None, :]

    def correction(bessel_argument: float, node_count: int) -> np.ndarray:
        hermite_nodes, hermite_weights = roots_hermite(node_count)
        laguerre_nodes, laguerre_weights = roots_laguerre(node_count)
        hermite_normalization = np.exp(
            0.5
            * (
                np.arange(maximum_hermite_order + 1) * np.log(2.0)
                + gammaln(np.arange(maximum_hermite_order + 1) + 1)
            )
        )
        hermite = np.asarray(
            [
                eval_hermite(order, hermite_nodes) / hermite_normalization[order]
                for order in range(maximum_hermite_order + 1)
            ]
        )
        laguerre = np.asarray(
            [eval_laguerre(order, laguerre_nodes) for order in range(n_laguerre)]
        )
        speed_squared = hermite_nodes[:, None] ** 2 + laguerre_nodes[None, :]
        generalized = np.asarray(
            [
                eval_genlaguerre(order, 1.5, speed_squared)
                for order in range(correction_order + 1)
            ]
        )
        weights = hermite_weights[:, None] * laguerre_weights[None, :] / np.sqrt(np.pi)
        argument = bessel_argument * np.sqrt(laguerre_nodes)
        parallel = np.einsum(
            "ph,jx,khx,hx,x,h->pjk",
            hermite,
            laguerre,
            generalized,
            weights,
            jv(0, argument),
            hermite_nodes,
            optimize=True,
        ).reshape(-1, correction_order + 1)
        perpendicular = np.einsum(
            "ph,jx,khx,hx,x->pjk",
            hermite,
            laguerre,
            generalized,
            weights,
            jv(1, argument) * np.sqrt(laguerre_nodes),
            optimize=True,
        ).reshape(-1, correction_order + 1)
        return (
            parallel @ correction_kernel @ parallel.T
            + perpendicular @ correction_kernel @ perpendicular.T
        )

    field = original_field.copy()
    for wavelength_index, bessel_argument in enumerate(grid):
        if bessel_argument == 0.0:
            # Preserve the exact drift-kinetic limit instead of allowing the
            # independent quadrature route to populate zero invariant entries
            # at a SciPy/BLAS-dependent roundoff level.
            _paper_test, paper_field = (
                improved_sugama_equal_temperature_moment_matrices(
                    convention * np.asarray(coulomb_test_table[wavelength_index]),
                    maximum_hermite_order,
                    maximum_laguerre_order,
                    correction_order=correction_order,
                    digits=digits,
                )
            )
            field[wavelength_index] = convention * paper_field
            continue
        lower = correction(float(bessel_argument), quadrature_order)
        accepted = correction(float(bessel_argument), quadrature_order + 16)
        difference = float(np.linalg.norm(accepted - lower))
        scale = max(float(np.linalg.norm(accepted)), np.finfo(float).tiny)
        if difference > 1.0e-12 and difference / scale > 1.0e-11:
            raise RuntimeError("improved-Sugama velocity quadrature did not converge")
        field[wavelength_index] += convention * accepted
    return original_test, field


def coulomb_polarization_vectors(
    maximum_hermite_order: int,
    maximum_laguerre_order: int,
    target_bessel_argument: float,
    source_bessel_argument: float,
    mass_ratio: float,
    temperature_ratio: float,
    *,
    maximum_spherical_order: int | None = None,
    maximum_spherical_radial_order: int | None = None,
    maximum_angular_bessel_order: int | None = None,
    included_angular_orders: tuple[int, ...] | None = None,
    maximum_bessel_laguerre_order: int = 24,
    digits: int = 80,
    float64_final_contraction: bool = False,
    worker_count: int = 1,
    _coefficient_functions: tuple[Callable[..., Any], ...] | None = None,
    _assembly_cache: dict[str, dict[tuple[Any, ...], Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Generate the four Coulomb polarization vectors in equation (3.50).

    The result is ``(test_phi1, field_phi1, test_phi2, field_phi2)`` in the
    paper's Laguerre convention.  Test vectors multiply ``q_a phi / T_a``;
    field vectors multiply ``q_b phi / T_b``.  Target and source gyroradii are
    separate because unlike-species polarization uses both.
    ``included_angular_orders`` selects a sorted subset for resumable offline
    generation; only the order-zero block owns the ``phi1`` contribution.
    On the float64 archive path, ``worker_count > 1`` partitions the additive
    equation-(3.50) contraction by independent angular harmonic ``m``.
    """

    if maximum_hermite_order < 0:
        raise ValueError("maximum_hermite_order must be >= 0")
    if maximum_laguerre_order < 0:
        raise ValueError("maximum_laguerre_order must be >= 0")
    for value, name in (
        (target_bessel_argument, "target_bessel_argument"),
        (source_bessel_argument, "source_bessel_argument"),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and >= 0")
    if mass_ratio <= 0.0 or not math.isfinite(mass_ratio):
        raise ValueError("mass_ratio must be finite and > 0")
    if temperature_ratio <= 0.0 or not math.isfinite(temperature_ratio):
        raise ValueError("temperature_ratio must be finite and > 0")
    if maximum_bessel_laguerre_order < 0:
        raise ValueError("maximum_bessel_laguerre_order must be >= 0")
    if maximum_angular_bessel_order is not None and maximum_angular_bessel_order < 0:
        raise ValueError("maximum_angular_bessel_order must be >= 0")
    if digits < 16:
        raise ValueError("digits must be >= 16")
    if worker_count < 1:
        raise ValueError("worker_count must be >= 1")

    maximum_degree = maximum_hermite_order + 2 * maximum_laguerre_order
    spherical_limit = (
        maximum_degree if maximum_spherical_order is None else maximum_spherical_order
    )
    radial_limit = (
        maximum_degree // 2
        if maximum_spherical_radial_order is None
        else maximum_spherical_radial_order
    )
    if spherical_limit < 0:
        raise ValueError("maximum_spherical_order must be >= 0")
    if radial_limit < 0:
        raise ValueError("maximum_spherical_radial_order must be >= 0")
    angular_limit = (
        spherical_limit
        if maximum_angular_bessel_order is None
        else min(spherical_limit, maximum_angular_bessel_order)
    )
    active_angular_orders = _resolve_collision_angular_orders(
        angular_limit,
        included_angular_orders,
        drift_kinetic=(target_bessel_argument == 0.0),
    )

    import mpmath as mp

    n_laguerre = maximum_laguerre_order + 1
    n_modes = (maximum_hermite_order + 1) * n_laguerre
    with mp.workdps(digits):
        target_b = mp.mpf(target_bessel_argument)
        source_b = mp.mpf(source_bessel_argument)
        sigma = mp.mpf(mass_ratio)
        tau = mp.mpf(temperature_ratio)
        half_target_b = target_b / 2
        target_argument = half_target_b * half_target_b
        test_phi1 = mp.matrix(n_modes, 1)
        field_phi1 = mp.matrix(n_modes, 1)
        test_phi2 = mp.matrix(n_modes, 1)
        field_phi2 = mp.matrix(n_modes, 1)
        assembly_cache = {} if _assembly_cache is None else _assembly_cache
        speed_cache = assembly_cache.setdefault("integrated_speed", {})
        polarization_cache = assembly_cache.setdefault("polarization", {})
        polarization_bessel_cache = assembly_cache.setdefault(
            "polarization_bessel_kernel", {}
        )
        polarization_radial_cache = assembly_cache.setdefault(
            "polarization_radial_kernel", {}
        )
        product_cache = assembly_cache.setdefault("laguerre_product", {})
        inverse_cache = assembly_cache.setdefault("inverse_transform", {})
        coefficient_functions = (
            _coulomb_coefficient_functions(mp, sigma, tau)
            if _coefficient_functions is None
            else _coefficient_functions
        )
        (
            _base_transform,
            associated_laguerre,
            _legendre_monomial,
            inverse_associated,
            inverse_product,
            speed_moment,
            associated_spherical_polynomial,
            hermite_gaussian_moment,
            laguerre_exponential_moment,
        ) = coefficient_functions

        def integrated_speed(p: int, j: int, t: int) -> tuple[Any, Any]:
            key = (p, j, t)
            if key not in speed_cache:
                test_speed = mp.mpf(0)
                field_speed = mp.mpf(0)
                for speed_power in range(t + 1):
                    coefficient = associated_laguerre(
                        t,
                        p,
                        speed_power,
                    )
                    test_term, field_term = speed_moment(p, j, speed_power)
                    test_speed += coefficient * test_term
                    field_speed += coefficient * field_term
                speed_cache[key] = (test_speed, field_speed)
            return speed_cache[key]

        def polarization(p: int, j: int, m: int, *, source: bool) -> Any:
            wavelength = source_b if source else target_b
            key = (float(wavelength), p, j, m)
            if key not in polarization_cache:
                bessel_key = (
                    float(wavelength),
                    m,
                    maximum_bessel_laguerre_order,
                )
                if bessel_key not in polarization_bessel_cache:
                    polarization_bessel_cache[bessel_key] = _bessel_laguerre_kernels_mp(
                        wavelength,
                        m,
                        maximum_bessel_laguerre_order + 1,
                        mp,
                    )
                radial_count = (
                    2 * maximum_bessel_laguerre_order
                    + 2 * spherical_limit
                    + radial_limit
                    + 1
                )
                radial_key = (float(wavelength), radial_count)
                if radial_key not in polarization_radial_cache:
                    polarization_radial_cache[radial_key] = _radial_poisson_kernels_mp(
                        wavelength,
                        radial_count,
                        mp,
                    )
                polarization_cache[key] = _gyroaveraged_polarization_coefficient_mp(
                    p,
                    j,
                    m,
                    wavelength,
                    maximum_bessel_laguerre_order,
                    mp,
                    associated_transform=inverse_associated,
                    laguerre_product=inverse_product,
                    bessel_kernels=polarization_bessel_cache[bessel_key],
                    radial_kernels=polarization_radial_cache[radial_key],
                    float64_final_contraction=float64_final_contraction,
                )
            return polarization_cache[key]

        def laguerre_product(
            m: int,
            n: int,
            output_laguerre: int,
            product_order: int,
        ) -> Any:
            key = (m, n, output_laguerre, product_order)
            if key not in product_cache:
                product_cache[key] = inverse_product(
                    m,
                    n,
                    output_laguerre,
                    product_order,
                    0,
                )
            return product_cache[key]

        def inverse_transform(
            output_hermite: int,
            product_order: int,
            p: int,
            speed_order: int,
            m: int,
        ) -> Any:
            key = (output_hermite, product_order, p, speed_order, m)
            if key not in inverse_cache:
                inverse_cache[key] = _hermite_laguerre_to_associated_legendre_mp(
                    output_hermite,
                    product_order,
                    p,
                    speed_order,
                    m,
                    mp,
                    associated_spherical_polynomial=associated_spherical_polynomial,
                    hermite_gaussian_moment=hermite_gaussian_moment,
                    laguerre_exponential_moment=laguerre_exponential_moment,
                )
            return inverse_cache[key]

        bessel_orders = tuple(range(maximum_bessel_laguerre_order + 1))
        exponential = mp.exp(-target_argument)
        bessel_factors = {
            m: tuple(
                exponential
                * target_argument**n
                * half_target_b**m
                / mp.factorial(n + m)
                for n in bessel_orders
            )
            for m in active_angular_orders
        }
        weighted_products = {
            (m, output_laguerre): _bessel_weighted_laguerre_products_mp(
                m,
                output_laguerre,
                bessel_orders,
                bessel_factors[m],
                laguerre_product,
                mp,
            )
            for m in active_angular_orders
            for output_laguerre in range(n_laguerre)
        }
        if float64_final_contraction and worker_count > 1:
            angular_tasks: list[tuple[int, tuple[int, ...], bool]] = []
            if len(active_angular_orders) == 1:
                angular_order = active_angular_orders[0]
                p_values = tuple(range(angular_order, spherical_limit + 1))
                task_count = min(worker_count, len(p_values))
                angular_tasks.extend(
                    (
                        angular_order,
                        tuple(p_values[index::task_count]),
                        angular_order == 0 and index == 0,
                    )
                    for index in range(task_count)
                )
            else:
                angular_tasks.extend(
                    (
                        angular_order,
                        tuple(range(angular_order, spherical_limit + 1)),
                        angular_order == 0,
                    )
                    for angular_order in active_angular_orders
                )

            def build_angular_contribution(
                task_index: int,
            ) -> tuple[int, tuple[np.ndarray, ...]]:
                angular_order, p_values, owns_phi1 = angular_tasks[task_index]
                local_vectors = [mp.matrix(n_modes, 1) for _ in range(4)]
                radial_groups: list[tuple[int, tuple[tuple[Any, ...], ...]]] = []
                for p in p_values:
                    entries = []
                    for j in range(radial_limit + 1):
                        sigma_pj = (
                            mp.factorial(p)
                            * mp.gamma(p + j + mp.mpf("1.5"))
                            / (
                                mp.power(2, p)
                                * mp.gamma(p + mp.mpf("1.5"))
                                * mp.factorial(j)
                            )
                        )
                        angular = (
                            (1 if angular_order == 0 else 2)
                            * mp.power(2, p)
                            * mp.factorial(p) ** 2
                            / (sigma_pj * mp.factorial(2 * p) * (2 * p + 1))
                        )
                        target_value = polarization(p, j, angular_order, source=False)
                        source_value = polarization(p, j, angular_order, source=True)
                        if target_value != 0 or source_value != 0:
                            entries.append((j, target_value, source_value, angular))
                    if entries:
                        radial_groups.append((p, tuple(entries)))

                for output_hermite in range(maximum_hermite_order + 1):
                    output_normalization = mp.sqrt(
                        mp.power(2, output_hermite) * mp.factorial(output_hermite)
                    )
                    for output_laguerre in range(n_laguerre):
                        row = output_hermite * n_laguerre + output_laguerre
                        if owns_phi1:
                            for product_order, weighted_product in enumerate(
                                weighted_products[(0, output_laguerre)]
                            ):
                                if weighted_product == 0:
                                    continue
                                for speed_order in range(
                                    product_order + output_hermite // 2 + 1
                                ):
                                    inverse = inverse_transform(
                                        output_hermite,
                                        product_order,
                                        0,
                                        speed_order,
                                        0,
                                    )
                                    if inverse == 0:
                                        continue
                                    test_speed, field_speed = integrated_speed(
                                        0, 0, speed_order
                                    )
                                    common = (
                                        weighted_product
                                        * inverse
                                        / output_normalization
                                    )
                                    local_vectors[0][row] -= common * test_speed
                                    local_vectors[1][row] -= common * field_speed

                        for p, radial_moments in radial_groups:
                            speed_weights: dict[int, Any] = {}
                            for product_order, weighted_product in enumerate(
                                weighted_products[(angular_order, output_laguerre)]
                            ):
                                if (
                                    weighted_product == 0
                                    or p
                                    > output_hermite + angular_order + 2 * product_order
                                ):
                                    continue
                                maximum_speed_order = (
                                    product_order
                                    + (output_hermite + angular_order) // 2
                                )
                                for speed_order in range(maximum_speed_order + 1):
                                    inverse = inverse_transform(
                                        output_hermite,
                                        product_order,
                                        p,
                                        speed_order,
                                        angular_order,
                                    )
                                    if inverse != 0:
                                        speed_weights[speed_order] = (
                                            speed_weights.get(speed_order, mp.mpf(0))
                                            + weighted_product
                                            * inverse
                                            / output_normalization
                                        )
                            for (
                                j,
                                target_value,
                                source_value,
                                angular,
                            ) in radial_moments:
                                test_coefficient = mp.mpf(0)
                                field_coefficient = mp.mpf(0)
                                for speed_order, weight in speed_weights.items():
                                    test_speed, field_speed = integrated_speed(
                                        p, j, speed_order
                                    )
                                    test_coefficient += weight * test_speed
                                    field_coefficient += weight * field_speed
                                local_vectors[2][row] += (
                                    angular * target_value * test_coefficient
                                )
                                local_vectors[3][row] += (
                                    angular * source_value * field_coefficient
                                )
                return task_index, tuple(
                    np.asarray(vector.tolist(), dtype=np.float64).reshape(n_modes)
                    for vector in local_vectors
                )

            contributions = _forked_ordered_map(
                build_angular_contribution,
                len(angular_tasks),
                min(worker_count, len(angular_tasks)),
            )
            vectors = [np.zeros(n_modes, dtype=np.float64) for _ in range(4)]
            for _angular_order, components in contributions:
                for vector, component in zip(vectors, components, strict=True):
                    vector += component
            return tuple(vectors)  # type: ignore[return-value]

        grouped_polarization: dict[tuple[int, int], list[tuple[Any, ...]]] = {}
        for p in range(spherical_limit + 1):
            for j in range(radial_limit + 1):
                sigma_pj = (
                    mp.factorial(p)
                    * mp.gamma(p + j + mp.mpf("1.5"))
                    / (mp.power(2, p) * mp.gamma(p + mp.mpf("1.5")) * mp.factorial(j))
                )
                angular_base = (
                    mp.power(2, p)
                    * mp.factorial(p) ** 2
                    / (sigma_pj * mp.factorial(2 * p) * (2 * p + 1))
                )
                for m in active_angular_orders:
                    if m > p:
                        continue
                    target_polarization = polarization(p, j, m, source=False)
                    source_polarization = polarization(p, j, m, source=True)
                    if target_polarization == 0 and source_polarization == 0:
                        continue
                    grouped_polarization.setdefault((p, m), []).append(
                        (
                            j,
                            target_polarization,
                            source_polarization,
                            (1 if m == 0 else 2) * angular_base,
                        )
                    )
        active_polarization_groups = tuple(
            (p, m, tuple(entries)) for (p, m), entries in grouped_polarization.items()
        )

        for output_hermite in range(maximum_hermite_order + 1):
            output_normalization = mp.sqrt(
                mp.power(2, output_hermite) * mp.factorial(output_hermite)
            )
            for output_laguerre in range(n_laguerre):
                row = output_hermite * n_laguerre + output_laguerre
                if 0 in active_angular_orders:
                    for product_order, weighted_product in enumerate(
                        weighted_products[(0, output_laguerre)]
                    ):
                        if weighted_product == 0:
                            continue
                        for speed_order in range(
                            product_order + output_hermite // 2 + 1
                        ):
                            inverse = inverse_transform(
                                output_hermite,
                                product_order,
                                0,
                                speed_order,
                                0,
                            )
                            if inverse == 0:
                                continue
                            test_speed, field_speed = integrated_speed(
                                0,
                                0,
                                speed_order,
                            )
                            common = weighted_product * inverse / output_normalization
                            test_phi1[row] -= common * test_speed
                            field_phi1[row] -= common * field_speed

                for p, m, radial_moments in active_polarization_groups:
                    speed_weights: dict[int, Any] = {}
                    for product_order, weighted_product in enumerate(
                        weighted_products[(m, output_laguerre)]
                    ):
                        if (
                            weighted_product == 0
                            or p > output_hermite + m + 2 * product_order
                        ):
                            continue
                        maximum_speed_order = product_order + (output_hermite + m) // 2
                        for speed_order in range(maximum_speed_order + 1):
                            inverse = inverse_transform(
                                output_hermite,
                                product_order,
                                p,
                                speed_order,
                                m,
                            )
                            if inverse == 0:
                                continue
                            common = weighted_product * inverse / output_normalization
                            speed_weights[speed_order] = (
                                speed_weights.get(
                                    speed_order,
                                    mp.mpf(0),
                                )
                                + common
                            )
                    for (
                        j,
                        target_polarization,
                        source_polarization,
                        angular,
                    ) in radial_moments:
                        test_coefficient = mp.mpf(0)
                        field_coefficient = mp.mpf(0)
                        for speed_order, weight in speed_weights.items():
                            test_speed, field_speed = integrated_speed(
                                p,
                                j,
                                speed_order,
                            )
                            test_coefficient += weight * test_speed
                            field_coefficient += weight * field_speed
                        test_phi2[row] += (
                            angular * target_polarization * test_coefficient
                        )
                        field_phi2[row] += (
                            angular * source_polarization * field_coefficient
                        )

        vectors = tuple(
            np.asarray(vector.tolist(), dtype=np.float64).reshape(n_modes)
            for vector in (test_phi1, field_phi1, test_phi2, field_phi2)
        )
    return vectors


def build_finite_wavelength_coulomb_pair_tables(
    bessel_arguments: tuple[float, ...],
    maximum_hermite_order: int,
    maximum_laguerre_order: int,
    mass_ratio: float,
    temperature_ratio: float,
    *,
    maximum_spherical_order: int | None = None,
    maximum_spherical_radial_order: int | None = None,
    maximum_angular_bessel_order: int | None = None,
    maximum_bessel_laguerre_order: int = 24,
    digits: int = 80,
    worker_count: int = 1,
) -> tuple[np.ndarray, ...]:
    r"""Build one ordered-pair table for the JAX finite-wavelength operator.

    The returned test/field matrices and four polarization vectors have
    independent target/source :math:`B=k_\perp v_{\mathrm{th}}/\Omega` axes.
    All wavelength-independent multiprecision basis algebra is shared across
    the scan. Unlike the two
    equation-level generators, these tables use the runtime's signed Laguerre
    convention and can therefore be inserted directly below target/source
    species axes in :class:`FiniteWavelengthCoulombOperator`. Polarization is
    assembled first so forked matrix rows inherit its exact transform caches.
    """

    grid = np.asarray(bessel_arguments, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("bessel_arguments must contain at least two points")
    if not np.all(np.isfinite(grid)) or np.any(grid < 0.0):
        raise ValueError("bessel_arguments must be finite and >= 0")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("bessel_arguments must be strictly increasing")
    if worker_count < 1:
        raise ValueError("worker_count must be >= 1")

    import mpmath as mp

    mode_count = (maximum_hermite_order + 1) * (maximum_laguerre_order + 1)
    matrices = [
        np.empty((grid.size, grid.size, mode_count, mode_count)) for _ in range(2)
    ]
    vectors = [np.empty((grid.size, grid.size, mode_count)) for _ in range(4)]
    laguerre_sign = np.asarray(
        [
            (-1.0) ** laguerre_order
            for _hermite_order in range(maximum_hermite_order + 1)
            for laguerre_order in range(maximum_laguerre_order + 1)
        ]
    )
    matrix_convention = laguerre_sign[:, None] * laguerre_sign[None, :]

    with mp.workdps(digits):
        coefficient_functions = _coulomb_coefficient_functions(
            mp,
            mp.mpf(mass_ratio),
            mp.mpf(temperature_ratio),
        )
        assembly_cache: dict[str, dict[tuple[Any, ...], Any]] = {}
        for target_index, target_argument in enumerate(grid):
            for source_index, source_argument in enumerate(grid):
                pair_vectors = coulomb_polarization_vectors(
                    maximum_hermite_order,
                    maximum_laguerre_order,
                    float(target_argument),
                    float(source_argument),
                    mass_ratio,
                    temperature_ratio,
                    maximum_spherical_order=maximum_spherical_order,
                    maximum_spherical_radial_order=maximum_spherical_radial_order,
                    maximum_angular_bessel_order=maximum_angular_bessel_order,
                    maximum_bessel_laguerre_order=maximum_bessel_laguerre_order,
                    digits=digits,
                    _coefficient_functions=coefficient_functions,
                    _assembly_cache=assembly_cache,
                )
                pair_matrices = coulomb_nonpolarized_moment_matrices(
                    maximum_hermite_order,
                    maximum_laguerre_order,
                    float(target_argument),
                    mass_ratio,
                    temperature_ratio,
                    source_bessel_argument=float(source_argument),
                    maximum_spherical_order=maximum_spherical_order,
                    maximum_spherical_radial_order=maximum_spherical_radial_order,
                    maximum_angular_bessel_order=maximum_angular_bessel_order,
                    maximum_bessel_laguerre_order=maximum_bessel_laguerre_order,
                    digits=digits,
                    worker_count=worker_count,
                    _coefficient_functions=coefficient_functions,
                    _assembly_cache=assembly_cache,
                )
                for table, values in zip(matrices, pair_matrices, strict=True):
                    table[target_index, source_index] = matrix_convention * values
                for table, values in zip(vectors, pair_vectors, strict=True):
                    table[target_index, source_index] = laguerre_sign * values
    return (*matrices, *vectors)


def write_finite_wavelength_coulomb_endpoint(
    out: Path,
    *,
    bessel_argument: float,
    maximum_hermite_order: int,
    maximum_laguerre_order: int,
    maximum_angular_bessel_order: int,
    maximum_bessel_laguerre_order: int,
    digits: int = 32,
    worker_count: int = 1,
) -> dict[str, Any]:
    """Write one equal-species Coulomb endpoint with shared exact caches.

    The archive stores the two matrices and four polarization vectors in the
    paper's Laguerre convention. It is intentionally a fixed-wavelength
    research artifact, not an interpolation table.
    """

    import mpmath as mp

    maximum_degree = maximum_hermite_order + 2 * maximum_laguerre_order
    with mp.workdps(digits):
        started = time.perf_counter()
        print("collision endpoint: preparing speed coefficients", flush=True)
        coefficient_functions = _coulomb_coefficient_functions(mp, mp.mpf(1), mp.mpf(1))
        coefficient_functions = _precompute_coulomb_speed_coefficients(
            coefficient_functions,
            maximum_spherical_order=maximum_degree,
            maximum_spherical_radial_order=maximum_degree // 2,
            maximum_speed_power=(
                maximum_laguerre_order
                + maximum_bessel_laguerre_order
                + (maximum_hermite_order + maximum_angular_bessel_order) // 2
            ),
            worker_count=worker_count,
        )
        speed_precompute_seconds = time.perf_counter() - started
        assembly_cache: dict[str, dict[tuple[Any, ...], Any]] = {}
        kwargs = {
            "maximum_spherical_order": maximum_degree,
            "maximum_spherical_radial_order": maximum_degree // 2,
            "maximum_angular_bessel_order": maximum_angular_bessel_order,
            "maximum_bessel_laguerre_order": maximum_bessel_laguerre_order,
            "digits": digits,
            "_coefficient_functions": coefficient_functions,
            "_assembly_cache": assembly_cache,
        }
        started = time.perf_counter()
        print("collision endpoint: assembling polarization vectors", flush=True)
        vectors = coulomb_polarization_vectors(
            maximum_hermite_order,
            maximum_laguerre_order,
            bessel_argument,
            bessel_argument,
            1.0,
            1.0,
            **kwargs,
            float64_final_contraction=True,
            worker_count=worker_count,
        )
        polarization_seconds = time.perf_counter() - started
        started = time.perf_counter()
        print("collision endpoint: assembling test/field matrices", flush=True)
        matrices = coulomb_nonpolarized_moment_matrices(
            maximum_hermite_order,
            maximum_laguerre_order,
            bessel_argument,
            1.0,
            1.0,
            **kwargs,
            float64_final_contraction=True,
            worker_count=worker_count,
        )
        matrix_seconds = time.perf_counter() - started

    arrays = (*matrices, *vectors)
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise RuntimeError("generated Coulomb endpoint contains non-finite values")
    checksum = float(sum(float(np.sum(array)) for array in arrays))
    metadata = {
        "schema_version": 1,
        "claim_scope": "fixed_wavelength_equal_species_coulomb_coefficients",
        "resolution": [maximum_hermite_order, maximum_laguerre_order],
        "bessel_argument": float(bessel_argument),
        "paper_kperp_at_tau_one": float(bessel_argument / np.sqrt(2.0)),
        "maximum_angular_bessel_order": maximum_angular_bessel_order,
        "maximum_bessel_laguerre_order": maximum_bessel_laguerre_order,
        "precision_decimal_digits": digits,
        "worker_count": worker_count,
        "float64_final_contraction": True,
        "speed_precompute_seconds": speed_precompute_seconds,
        "polarization_seconds": polarization_seconds,
        "matrix_seconds": matrix_seconds,
        "total_seconds": (
            speed_precompute_seconds + polarization_seconds + matrix_seconds
        ),
        "checksum": checksum,
        "laguerre_convention": "paper_unsigned",
        "source": "Frei et al. (2021), equations (3.48)--(3.50)",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        **{f"array_{index}": array for index, array in enumerate(arrays)},
    )
    return metadata


def write_equal_species_finite_wavelength_coulomb_table(
    out: Path,
    *,
    bessel_arguments: tuple[float, ...],
    maximum_hermite_order: int,
    maximum_laguerre_order: int,
    maximum_angular_bessel_order: int,
    maximum_bessel_laguerre_order: int,
    included_angular_orders: tuple[int, ...] | None = None,
    digits: int = 32,
    worker_count: int = 1,
    wavelength_worker_count: int = 1,
    _coefficient_functions: tuple[Callable[..., Any], ...] | None = None,
) -> dict[str, Any]:
    """Write the diagonal finite-wavelength table needed by one ion species.

    Target and source Bessel arguments are equal pointwise for like-species
    collisions. All wavelengths share the expensive speed coefficients. The
    default serial wavelength route also shares algebra caches; the opt-in
    wavelength decomposition gives each point a private cache so expensive
    high-order points can run concurrently. Stored coefficients use the
    runtime's signed Laguerre convention and can be passed directly to
    ``EqualSpeciesFiniteWavelengthCoulombOperator``.
    """

    import mpmath as mp

    grid = np.asarray(bessel_arguments, dtype=float)
    minimum_grid_size = 1 if included_angular_orders is not None else 2
    if grid.ndim != 1 or grid.size < minimum_grid_size:
        raise ValueError(
            f"bessel_arguments must contain at least {minimum_grid_size} point(s)"
        )
    if np.any(~np.isfinite(grid)) or np.any(grid < 0.0) or np.any(np.diff(grid) <= 0.0):
        raise ValueError(
            "bessel_arguments must be finite, nonnegative, and strictly increasing"
        )
    if worker_count < 1 or wavelength_worker_count < 1:
        raise ValueError("worker counts must be >= 1")
    maximum_degree = maximum_hermite_order + 2 * maximum_laguerre_order
    mode_count = (maximum_hermite_order + 1) * (maximum_laguerre_order + 1)
    matrices = [np.empty((grid.size, mode_count, mode_count)) for _ in range(2)]
    vectors = [np.empty((grid.size, mode_count)) for _ in range(4)]
    laguerre_sign = np.asarray(
        [
            (-1.0) ** laguerre_order
            for _hermite_order in range(maximum_hermite_order + 1)
            for laguerre_order in range(maximum_laguerre_order + 1)
        ]
    )
    matrix_convention = laguerre_sign[:, None] * laguerre_sign[None, :]
    active_angular_orders = _resolve_collision_angular_orders(
        maximum_angular_bessel_order,
        included_angular_orders,
        drift_kinetic=False,
    )

    with mp.workdps(digits):
        total_started = time.perf_counter()
        if _coefficient_functions is None:
            print(
                "collision table: preparing wavelength-independent coefficients",
                flush=True,
            )
            coefficient_functions = _coulomb_coefficient_functions(
                mp, mp.mpf(1), mp.mpf(1)
            )
            coefficient_functions = _precompute_coulomb_speed_coefficients(
                coefficient_functions,
                maximum_spherical_order=maximum_degree,
                maximum_spherical_radial_order=maximum_degree // 2,
                maximum_speed_power=(
                    maximum_laguerre_order
                    + maximum_bessel_laguerre_order
                    + (maximum_hermite_order + maximum_angular_bessel_order) // 2
                ),
                worker_count=worker_count,
            )
        else:
            coefficient_functions = _coefficient_functions
        speed_precompute_seconds = time.perf_counter() - total_started
        active_wavelength_workers = min(wavelength_worker_count, grid.size)
        inner_worker_count = max(1, worker_count // active_wavelength_workers)

        def build_wavelength(
            index: int,
            assembly_cache: dict[str, dict[tuple[Any, ...], Any]],
            point_worker_count: int,
        ) -> tuple[int, tuple[np.ndarray, ...], float]:
            bessel_argument = float(grid[index])
            started = time.perf_counter()
            print(
                f"collision table: wavelength {index + 1}/{grid.size}, B={bessel_argument:.9g}",
                flush=True,
            )
            kwargs = {
                "maximum_spherical_order": maximum_degree,
                "maximum_spherical_radial_order": maximum_degree // 2,
                "maximum_angular_bessel_order": maximum_angular_bessel_order,
                "included_angular_orders": active_angular_orders,
                "maximum_bessel_laguerre_order": maximum_bessel_laguerre_order,
                "digits": digits,
                "_coefficient_functions": coefficient_functions,
                "_assembly_cache": assembly_cache,
            }
            point_vectors = coulomb_polarization_vectors(
                maximum_hermite_order,
                maximum_laguerre_order,
                float(bessel_argument),
                float(bessel_argument),
                1.0,
                1.0,
                **kwargs,
                float64_final_contraction=True,
                worker_count=point_worker_count,
            )
            point_matrices = coulomb_nonpolarized_moment_matrices(
                maximum_hermite_order,
                maximum_laguerre_order,
                float(bessel_argument),
                1.0,
                1.0,
                **kwargs,
                float64_final_contraction=True,
                worker_count=point_worker_count,
            )
            return (
                index,
                (*point_matrices, *point_vectors),
                time.perf_counter() - started,
            )

        if active_wavelength_workers == 1:
            shared_cache: dict[str, dict[tuple[Any, ...], Any]] = {}
            point_results = [
                build_wavelength(index, shared_cache, worker_count)
                for index in range(grid.size)
            ]
        else:

            def build_parallel_wavelength(
                index: int,
            ) -> tuple[int, tuple[np.ndarray, ...], float]:
                return build_wavelength(index, {}, inner_worker_count)

            point_results = _forked_ordered_map(
                build_parallel_wavelength,
                grid.size,
                active_wavelength_workers,
            )

        wavelength_seconds = [0.0] * grid.size
        for index, point_arrays, elapsed_seconds in point_results:
            point_matrices = point_arrays[:2]
            point_vectors = point_arrays[2:]
            for table, values in zip(matrices, point_matrices, strict=True):
                table[index] = matrix_convention * values
            for table, values in zip(vectors, point_vectors, strict=True):
                table[index] = laguerre_sign * values
            wavelength_seconds[index] = elapsed_seconds
        total_seconds = time.perf_counter() - total_started

    arrays = (*matrices, *vectors)
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise RuntimeError("generated Coulomb table contains non-finite values")
    checksum = float(sum(float(np.sum(array)) for array in arrays))
    metadata = {
        "schema_version": 1,
        "claim_scope": "equal_species_diagonal_finite_wavelength_coulomb_table",
        "resolution": [maximum_hermite_order, maximum_laguerre_order],
        "bessel_argument_grid": grid.tolist(),
        "maximum_angular_bessel_order": maximum_angular_bessel_order,
        "included_angular_orders": list(active_angular_orders),
        "maximum_bessel_laguerre_order": maximum_bessel_laguerre_order,
        "precision_decimal_digits": digits,
        "worker_count": worker_count,
        "wavelength_worker_count": active_wavelength_workers,
        "workers_per_wavelength": inner_worker_count,
        "float64_final_contraction": True,
        "speed_precompute_seconds": speed_precompute_seconds,
        "wavelength_seconds": wavelength_seconds,
        "total_seconds": total_seconds,
        "checksum": checksum,
        "laguerre_convention": "runtime_signed",
        "source": "Frei et al. (2021), equations (3.48)--(3.50)",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        bessel_argument_grid=grid,
        test_table=matrices[0],
        field_table=matrices[1],
        test_phi1=vectors[0],
        field_phi1=vectors[1],
        test_phi2=vectors[2],
        field_phi2=vectors[3],
    )
    return metadata


def combine_equal_species_finite_wavelength_angular_shards(
    shard_paths: tuple[Path, ...],
    out: Path,
    *,
    shared_precompute_seconds: float | None = None,
    orchestration_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine complete single-harmonic table shards in angular order."""

    if not shard_paths:
        raise ValueError("at least one angular shard is required")
    array_names = (
        "test_table",
        "field_table",
        "test_phi1",
        "field_phi1",
        "test_phi2",
        "field_phi2",
    )
    records = []
    for path in shard_paths:
        with np.load(path) as archive:
            metadata = json.loads(str(archive["metadata"]))
            orders = tuple(metadata.get("included_angular_orders", ()))
            if len(orders) != 1:
                raise ValueError("each archive must contain exactly one angular order")
            arrays = tuple(np.asarray(archive[name]) for name in array_names)
        records.append((orders[0], metadata, arrays))
    records.sort(key=lambda record: record[0])
    first = records[0][1]
    maximum_order = int(first["maximum_angular_bessel_order"])
    if [record[0] for record in records] != list(range(maximum_order + 1)):
        raise ValueError("angular shards must provide complete contiguous coverage")
    identity_fields = (
        "resolution",
        "bessel_argument_grid",
        "maximum_angular_bessel_order",
        "maximum_bessel_laguerre_order",
        "precision_decimal_digits",
        "laguerre_convention",
        "source",
    )
    for _order, metadata, arrays in records:
        if any(metadata[field] != first[field] for field in identity_fields):
            raise ValueError("angular shard metadata mismatch")
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("angular shard contains non-finite values")
    combined = tuple(
        sum(
            (record[2][index] for record in records),
            np.zeros_like(records[0][2][index]),
        )
        for index in range(len(array_names))
    )
    metadata = {
        **{field: first[field] for field in identity_fields},
        "schema_version": 1,
        "claim_scope": "equal_species_diagonal_finite_wavelength_coulomb_table",
        "included_angular_orders": list(range(maximum_order + 1)),
        "float64_final_contraction": True,
        "angular_shard_checksums": [record[1]["checksum"] for record in records],
        "sum_shard_seconds": float(
            sum(record[1]["total_seconds"] for record in records)
        ),
        "maximum_shard_seconds": float(
            max(record[1]["total_seconds"] for record in records)
        ),
        "checksum": float(sum(float(np.sum(array)) for array in combined)),
    }
    if shared_precompute_seconds is not None:
        metadata["shared_precompute_seconds"] = float(shared_precompute_seconds)
    if orchestration_metadata is not None:
        metadata.update(orchestration_metadata)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        bessel_argument_grid=np.asarray(first["bessel_argument_grid"], dtype=float),
        **dict(zip(array_names, combined, strict=True)),
    )
    return metadata


def combine_equal_species_finite_wavelength_tables(
    table_paths: tuple[Path, ...],
    out: Path,
) -> dict[str, Any]:
    """Concatenate complete angular tables on a strictly ordered B grid."""

    if len(table_paths) < 2:
        raise ValueError("at least two wavelength tables are required")
    array_names = (
        "test_table",
        "field_table",
        "test_phi1",
        "field_phi1",
        "test_phi2",
        "field_phi2",
    )
    records = []
    for path in table_paths:
        with np.load(path) as archive:
            metadata = json.loads(str(archive["metadata"]))
            grid = np.asarray(archive["bessel_argument_grid"], dtype=float)
            arrays = tuple(np.asarray(archive[name]) for name in array_names)
        if grid.ndim != 1 or grid.size == 0:
            raise ValueError(
                "wavelength table grid must be one-dimensional and nonempty"
            )
        if np.any(~np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0):
            raise ValueError("each wavelength table grid must be strictly increasing")
        if metadata.get("bessel_argument_grid") != grid.tolist():
            raise ValueError("wavelength table grid metadata mismatch")
        if any(array.shape[0] != grid.size for array in arrays):
            raise ValueError("wavelength table array length does not match its grid")
        records.append((metadata, grid, arrays))
    first = records[0][0]
    identity_fields = (
        "resolution",
        "maximum_angular_bessel_order",
        "included_angular_orders",
        "maximum_bessel_laguerre_order",
        "precision_decimal_digits",
        "laguerre_convention",
        "source",
    )
    reference_shapes = tuple(array.shape[1:] for array in records[0][2])
    for metadata, _grid, arrays in records:
        if any(metadata[field] != first[field] for field in identity_fields):
            raise ValueError("wavelength table metadata mismatch")
        if metadata["included_angular_orders"] != list(
            range(int(metadata["maximum_angular_bessel_order"]) + 1)
        ):
            raise ValueError("wavelength tables must contain complete angular coverage")
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("wavelength table contains non-finite values")
        if tuple(array.shape[1:] for array in arrays) != reference_shapes:
            raise ValueError("wavelength table array shape mismatch")
    order = sorted(
        range(len(records)),
        key=lambda index: float(records[index][1][0]),
    )
    grid = np.concatenate([records[index][1] for index in order])
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("combined Bessel-argument grid must be strictly increasing")
    combined = tuple(
        np.concatenate([records[index][2][array_index] for index in order], axis=0)
        for array_index in range(len(array_names))
    )
    metadata = {
        **{field: first[field] for field in identity_fields},
        "schema_version": 1,
        "claim_scope": "equal_species_diagonal_finite_wavelength_coulomb_table",
        "bessel_argument_grid": grid.tolist(),
        "float64_final_contraction": True,
        "wavelength_table_checksums": [
            records[index][0]["checksum"] for index in order
        ],
        "checksum": float(sum(float(np.sum(array)) for array in combined)),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        bessel_argument_grid=grid,
        **dict(zip(array_names, combined, strict=True)),
    )
    return metadata


def project_equal_species_finite_wavelength_table(
    source: Path,
    out: Path,
    *,
    maximum_hermite_order: int,
    maximum_laguerre_order: int,
) -> dict[str, Any]:
    """Project a complete finite-wavelength archive onto a nested moment basis.

    Hermite-major mode ordering is preserved exactly: mode ``(p, j)`` has
    flattened index ``p * (J + 1) + j``. The operation is therefore the
    principal Galerkin projection of every matrix and polarization vector, not
    interpolation or coefficient regeneration.
    """

    if maximum_hermite_order < 0 or maximum_laguerre_order < 0:
        raise ValueError("target moment orders must be >= 0")
    with np.load(source) as archive:
        if "metadata" not in archive or "bessel_argument_grid" not in archive:
            raise ValueError("source must be a finite-wavelength table archive")
        metadata = json.loads(str(archive["metadata"]))
        arrays = {
            name: np.asarray(archive[name])
            for name in archive.files
            if name != "metadata"
        }
    claim_scope = str(metadata.get("claim_scope", ""))
    if not claim_scope.startswith("equal_species_diagonal_finite_wavelength_"):
        raise ValueError(
            "source must be a complete equal-species finite-wavelength table"
        )
    maximum_angular_order = int(metadata.get("maximum_angular_bessel_order", -1))
    if metadata.get("included_angular_orders") != list(
        range(maximum_angular_order + 1)
    ):
        raise ValueError("source table must contain complete angular coverage")
    if metadata.get("laguerre_convention") != "runtime_signed":
        raise ValueError("source table must use the runtime signed-Laguerre convention")
    grid = np.asarray(arrays["bessel_argument_grid"], dtype=float)
    if (
        grid.ndim != 1
        or grid.size < 2
        or np.any(~np.isfinite(grid))
        or np.any(np.diff(grid) <= 0.0)
        or metadata.get("bessel_argument_grid") != grid.tolist()
    ):
        raise ValueError("source wavelength grid and metadata must match")
    source_hermite, source_laguerre = map(int, metadata.get("resolution", ()))
    if (
        maximum_hermite_order > source_hermite
        or maximum_laguerre_order > source_laguerre
    ):
        raise ValueError("target resolution must not exceed the source resolution")
    source_mode_count = (source_hermite + 1) * (source_laguerre + 1)
    mode_indices = np.asarray(
        [
            hermite * (source_laguerre + 1) + laguerre
            for hermite in range(maximum_hermite_order + 1)
            for laguerre in range(maximum_laguerre_order + 1)
        ],
        dtype=int,
    )
    projected: dict[str, np.ndarray] = {}
    for name, values in arrays.items():
        if name == "bessel_argument_grid":
            projected[name] = values
        elif values.ndim >= 2 and values.shape[-2:] == (
            source_mode_count,
            source_mode_count,
        ):
            projected[name] = np.take(
                np.take(values, mode_indices, axis=-2), mode_indices, axis=-1
            )
        elif values.ndim >= 1 and values.shape[-1] == source_mode_count:
            projected[name] = np.take(values, mode_indices, axis=-1)
        else:
            raise ValueError(
                f"unsupported coefficient shape for {name}: {values.shape}"
            )
    coefficient_arrays = [
        values for name, values in projected.items() if name != "bessel_argument_grid"
    ]
    if not coefficient_arrays or any(
        not np.all(np.isfinite(values)) for values in coefficient_arrays
    ):
        raise ValueError("projected table must contain finite coefficients")
    output_metadata = {
        **metadata,
        "resolution": [maximum_hermite_order, maximum_laguerre_order],
        "derivation": "principal_hermite_laguerre_galerkin_projection",
        "parent_resolution": [source_hermite, source_laguerre],
        "parent_checksum": float(metadata["checksum"]),
        "checksum": float(sum(float(np.sum(values)) for values in coefficient_arrays)),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        metadata=np.asarray(json.dumps(output_metadata, sort_keys=True)),
        **projected,
    )
    return output_metadata


def _load_complete_equal_species_coulomb_table(
    coulomb_table: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Load and validate the Coulomb test data shared by Sugama conversions."""

    with np.load(coulomb_table) as archive:
        metadata = json.loads(str(archive["metadata"]))
        grid = np.asarray(archive["bessel_argument_grid"], dtype=float)
        test_table = np.asarray(archive["test_table"], dtype=float)
    if metadata.get("claim_scope") != (
        "equal_species_diagonal_finite_wavelength_coulomb_table"
    ):
        raise ValueError(
            "input must be an equal-species finite-wavelength Coulomb table"
        )
    maximum_order = int(metadata["maximum_angular_bessel_order"])
    if metadata.get("included_angular_orders") != list(range(maximum_order + 1)):
        raise ValueError("Coulomb table must contain complete angular coverage")
    if metadata.get("laguerre_convention") != "runtime_signed":
        raise ValueError(
            "Coulomb table must use the runtime signed-Laguerre convention"
        )
    return metadata, grid, test_table


def _write_equal_species_finite_wavelength_sugama_table(
    out: Path,
    *,
    coulomb_metadata: dict[str, Any],
    grid: np.ndarray,
    test: np.ndarray,
    field: np.ndarray,
    model: str,
    source: str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one validated original/improved equal-species Sugama archive."""

    maximum_order = int(coulomb_metadata["maximum_angular_bessel_order"])
    maximum_hermite_order, maximum_laguerre_order = map(
        int, coulomb_metadata["resolution"]
    )
    checksum = float(np.sum(test) + np.sum(field))
    output_metadata = {
        "schema_version": 1,
        "claim_scope": (
            f"equal_species_diagonal_finite_wavelength_{model}_sugama_table"
        ),
        "resolution": [maximum_hermite_order, maximum_laguerre_order],
        "bessel_argument_grid": grid.tolist(),
        "maximum_angular_bessel_order": maximum_order,
        "included_angular_orders": list(range(maximum_order + 1)),
        "maximum_bessel_laguerre_order": int(
            coulomb_metadata["maximum_bessel_laguerre_order"]
        ),
        "precision_decimal_digits": int(coulomb_metadata["precision_decimal_digits"]),
        "laguerre_convention": "runtime_signed",
        "coulomb_table_checksum": float(coulomb_metadata["checksum"]),
        "checksum": checksum,
        "source": source,
        **({} if extra_metadata is None else extra_metadata),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        metadata=np.asarray(json.dumps(output_metadata, sort_keys=True)),
        bessel_argument_grid=grid,
        test_table=test,
        field_table=field,
    )
    return output_metadata


def write_equal_species_finite_wavelength_original_sugama_table(
    coulomb_table: Path,
    out: Path,
    *,
    quadrature_order: int = 80,
) -> dict[str, Any]:
    """Convert a complete equal-species Coulomb test table to original Sugama."""

    metadata, grid, test_table = _load_complete_equal_species_coulomb_table(
        coulomb_table
    )
    maximum_hermite_order, maximum_laguerre_order = map(int, metadata["resolution"])
    test, field = finite_wavelength_original_sugama_like_species_moment_matrices(
        test_table,
        grid,
        maximum_hermite_order,
        maximum_laguerre_order,
        quadrature_order=quadrature_order,
    )
    return _write_equal_species_finite_wavelength_sugama_table(
        out,
        coulomb_metadata=metadata,
        grid=grid,
        test=test,
        field=field,
        model="original",
        source="Frei et al. (2021), equations (3.65), (3.68)--(3.69), and (3.79)--(3.80)",
        extra_metadata={
            "velocity_quadrature_orders": [
                int(quadrature_order),
                int(quadrature_order + 16),
            ],
            "quadrature_relative_tolerance": 1.0e-11,
            "quadrature_absolute_tolerance": 1.0e-12,
        },
    )


def write_equal_species_finite_wavelength_improved_sugama_table(
    coulomb_table: Path,
    out: Path,
    *,
    correction_order: int = 5,
    quadrature_order: int = 80,
    digits: int = 80,
) -> dict[str, Any]:
    """Convert a complete equal-species Coulomb test table to improved Sugama."""

    metadata, grid, test_table = _load_complete_equal_species_coulomb_table(
        coulomb_table
    )
    maximum_hermite_order, maximum_laguerre_order = map(int, metadata["resolution"])
    test, field = finite_wavelength_improved_sugama_like_species_moment_matrices(
        test_table,
        grid,
        maximum_hermite_order,
        maximum_laguerre_order,
        correction_order=correction_order,
        quadrature_order=quadrature_order,
        digits=digits,
    )
    return _write_equal_species_finite_wavelength_sugama_table(
        out,
        coulomb_metadata=metadata,
        grid=grid,
        test=test,
        field=field,
        model="improved",
        source="Frei, Ernst & Ricci (2022), equations (34)--(35) and (61)--(69)",
        extra_metadata={
            "improved_correction_order": int(correction_order),
            "velocity_quadrature_orders": [
                int(quadrature_order),
                int(quadrature_order + 16),
            ],
            "quadrature_relative_tolerance": 1.0e-11,
            "quadrature_absolute_tolerance": 1.0e-12,
        },
    )


def write_shared_precompute_angular_coulomb_table(
    out: Path,
    *,
    bessel_arguments: tuple[float, ...],
    maximum_hermite_order: int,
    maximum_laguerre_order: int,
    maximum_angular_bessel_order: int,
    maximum_bessel_laguerre_order: int,
    digits: int = 32,
    worker_count: int = 30,
    wavelength_worker_count: int = 2,
) -> dict[str, Any]:
    """Generate all angular shards from one shared speed-coefficient cache."""

    import mpmath as mp

    shard_count = maximum_angular_bessel_order + 1
    if worker_count < 1:
        raise ValueError("worker_count must be >= 1")
    maximum_degree = maximum_hermite_order + 2 * maximum_laguerre_order
    shard_paths = tuple(
        out.with_name(f"{out.stem}_m{order}{out.suffix}")
        for order in range(shard_count)
    )
    expected_grid = list(bessel_arguments)
    array_names = (
        "test_table",
        "field_table",
        "test_phi1",
        "field_phi1",
        "test_phi2",
        "field_phi2",
    )

    def reusable_metadata(order: int) -> dict[str, Any] | None:
        path = shard_paths[order]
        if not path.is_file():
            return None
        try:
            with np.load(path) as archive:
                metadata = json.loads(str(archive["metadata"]))
                finite = all(np.all(np.isfinite(archive[name])) for name in array_names)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None
        expected = {
            "resolution": [maximum_hermite_order, maximum_laguerre_order],
            "bessel_argument_grid": expected_grid,
            "maximum_angular_bessel_order": maximum_angular_bessel_order,
            "included_angular_orders": [order],
            "maximum_bessel_laguerre_order": maximum_bessel_laguerre_order,
            "precision_decimal_digits": digits,
            "laguerre_convention": "runtime_signed",
        }
        if not finite or any(
            metadata.get(key) != value for key, value in expected.items()
        ):
            return None
        return metadata

    retained_metadata = {
        order: metadata
        for order in range(shard_count)
        if (metadata := reusable_metadata(order)) is not None
    }
    pending_orders = tuple(
        order for order in range(shard_count) if order not in retained_metadata
    )
    if pending_orders and worker_count < len(pending_orders):
        raise ValueError(
            "worker_count must provide at least one worker per pending shard"
        )
    workers_per_shard = max(1, worker_count // max(1, len(pending_orders)))
    shared_precompute_seconds = 0.0
    generated_results: list[tuple[int, dict[str, Any]]] = []
    with mp.workdps(digits):
        if pending_orders:
            started = time.perf_counter()
            print("collision shards: preparing one shared speed cache", flush=True)
            coefficient_functions = _coulomb_coefficient_functions(
                mp, mp.mpf(1), mp.mpf(1)
            )
            coefficient_functions = _precompute_coulomb_speed_coefficients(
                coefficient_functions,
                maximum_spherical_order=maximum_degree,
                maximum_spherical_radial_order=maximum_degree // 2,
                maximum_speed_power=(
                    maximum_laguerre_order
                    + maximum_bessel_laguerre_order
                    + (maximum_hermite_order + maximum_angular_bessel_order) // 2
                ),
                worker_count=worker_count,
            )
            shared_precompute_seconds = time.perf_counter() - started

        def build_shard(pending_index: int) -> tuple[int, dict[str, Any]]:
            order = pending_orders[pending_index]
            metadata = write_equal_species_finite_wavelength_coulomb_table(
                shard_paths[order],
                bessel_arguments=bessel_arguments,
                maximum_hermite_order=maximum_hermite_order,
                maximum_laguerre_order=maximum_laguerre_order,
                maximum_angular_bessel_order=maximum_angular_bessel_order,
                maximum_bessel_laguerre_order=maximum_bessel_laguerre_order,
                included_angular_orders=(order,),
                digits=digits,
                worker_count=workers_per_shard,
                wavelength_worker_count=wavelength_worker_count,
                _coefficient_functions=coefficient_functions,
            )
            return order, metadata

        if pending_orders:
            generated_results = _forked_ordered_map(
                build_shard, len(pending_orders), len(pending_orders)
            )
    all_metadata = {**retained_metadata, **dict(generated_results)}
    combined = combine_equal_species_finite_wavelength_angular_shards(
        shard_paths,
        out,
        shared_precompute_seconds=shared_precompute_seconds,
        orchestration_metadata={
            "shard_worker_count": max(
                all_metadata[order]["worker_count"] for order in range(shard_count)
            ),
            "shard_worker_counts": [
                all_metadata[order]["worker_count"] for order in range(shard_count)
            ],
            "shard_total_seconds": [
                all_metadata[order]["total_seconds"] for order in range(shard_count)
            ],
            "reused_angular_orders": sorted(retained_metadata),
        },
    )
    return combined


def write_collision_table_contraction_gate(
    exact_archive: Path,
    fast_archive: Path,
    out_json: Path,
    *,
    maximum_relative_l2: float = 2.0e-14,
    maximum_absolute_error: float = 2.0e-13,
    minimum_speedup: float = 1.25,
) -> dict[str, Any]:
    """Gate a fast float64 final contraction against an exact table archive."""

    if min(maximum_relative_l2, maximum_absolute_error, minimum_speedup) <= 0.0:
        raise ValueError("contraction gate thresholds must be > 0")
    array_names = (
        "test_table",
        "field_table",
        "test_phi1",
        "field_phi1",
        "test_phi2",
        "field_phi2",
    )
    with np.load(exact_archive) as exact, np.load(fast_archive) as fast:
        exact_metadata = json.loads(str(exact["metadata"].item()))
        fast_metadata = json.loads(str(fast["metadata"].item()))
        for key in ("resolution", "bessel_argument_grid", "laguerre_convention"):
            if exact_metadata.get(key) != fast_metadata.get(key):
                raise ValueError(f"archive metadata mismatch: {key}")
        if not fast_metadata.get("float64_final_contraction", False):
            raise ValueError("fast archive does not declare float64 final contraction")
        arrays: dict[str, object] = {}
        coefficient_gate_passed = True
        for name in array_names:
            exact_values = np.asarray(exact[name], dtype=float)
            fast_values = np.asarray(fast[name], dtype=float)
            if exact_values.shape != fast_values.shape:
                raise ValueError(f"archive array shape mismatch: {name}")
            difference = fast_values - exact_values
            relative_l2 = float(
                np.linalg.norm(difference)
                / max(float(np.linalg.norm(exact_values)), 1.0e-300)
            )
            maximum_absolute = float(np.max(np.abs(difference)))
            passed = (
                relative_l2 <= maximum_relative_l2
                and maximum_absolute <= maximum_absolute_error
            )
            coefficient_gate_passed = coefficient_gate_passed and passed
            arrays[name] = {
                "relative_l2": relative_l2,
                "maximum_absolute_error": maximum_absolute,
                "bitwise_equal": bool(np.array_equal(exact_values, fast_values)),
                "passed": passed,
            }
    exact_seconds = float(exact_metadata["total_seconds"])
    fast_seconds = float(fast_metadata["total_seconds"])
    speedup = exact_seconds / fast_seconds
    speed_gate_passed = speedup >= minimum_speedup
    report = {
        "schema_version": 1,
        "claim_scope": "finite_wavelength_collision_table_final_contraction",
        "resolution": exact_metadata["resolution"],
        "bessel_argument_grid": exact_metadata["bessel_argument_grid"],
        "thresholds": {
            "maximum_relative_l2": maximum_relative_l2,
            "maximum_absolute_error": maximum_absolute_error,
            "minimum_speedup": minimum_speedup,
        },
        "timing_seconds": {"exact": exact_seconds, "fast": fast_seconds},
        "speedup": speedup,
        "arrays": arrays,
        "coefficient_gate_passed": coefficient_gate_passed,
        "speed_gate_passed": speed_gate_passed,
        "gate_passed": coefficient_gate_passed and speed_gate_passed,
        "notes": (
            "All collision coefficients are generated at the archive precision; "
            "only the final projection-vector contraction uses float64."
        ),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def build_coulomb_operator_verification_summary(*, digits: int = 80) -> dict[str, Any]:
    r"""Build equation, convergence, conservation, and entropy gates.

    This is an offline verification artifact for Frei et al. (2021), equations
    (3.41) and (3.48)--(3.50).  It deliberately stops short of transport claims:
    those require the generated blocks to be coupled through the runtime field
    solve and quasineutrality relation.
    """

    from scipy.special import eval_genlaguerre, jv, lpmv

    truncation_orders = np.asarray((0, 1, 2, 3, 4, 6, 8, 12), dtype=int)
    reference_order = 24
    truncation_values = np.asarray(
        [
            gyroaveraged_polarization_coefficient(
                3,
                1,
                1,
                1.8,
                maximum_bessel_laguerre_order=int(order),
                digits=digits,
            )
            for order in truncation_orders
        ]
    )
    reference_value = gyroaveraged_polarization_coefficient(
        3,
        1,
        1,
        1.8,
        maximum_bessel_laguerre_order=reference_order,
        digits=digits,
    )
    truncation_errors = np.abs(truncation_values - reference_value) / max(
        abs(reference_value), np.finfo(float).tiny
    )

    parallel, parallel_weights = np.polynomial.hermite.hermgauss(80)
    perpendicular, perpendicular_weights = np.polynomial.laguerre.laggauss(80)
    x_parallel = parallel[:, None]
    x_perpendicular = perpendicular[None, :]
    speed = np.sqrt(x_parallel**2 + x_perpendicular)
    pitch = x_parallel / speed
    projection_cases = (
        (0, 0, 0, 0.7),
        (1, 0, 1, 0.7),
        (2, 0, 0, 0.7),
        (2, 0, 2, 0.7),
        (3, 1, 1, 1.2),
    )
    projection_labels: list[str] = []
    projection_errors: list[float] = []
    for spherical_order, radial_order, bessel_order, b_value in projection_cases:
        generated = gyroaveraged_polarization_coefficient(
            spherical_order,
            radial_order,
            bessel_order,
            b_value,
            maximum_bessel_laguerre_order=14,
            digits=digits,
        )
        spherical_basis = (
            speed**spherical_order
            * lpmv(bessel_order, spherical_order, pitch)
            * eval_genlaguerre(
                radial_order,
                spherical_order + 0.5,
                speed**2,
            )
        )
        projected = np.sum(
            parallel_weights[:, None]
            * perpendicular_weights[None, :]
            * spherical_basis
            * jv(0, b_value * np.sqrt(x_perpendicular))
            * jv(bessel_order, b_value * np.sqrt(x_perpendicular))
        ) / np.sqrt(np.pi)
        projection_labels.append(
            rf"$\Pi^{{{spherical_order},{radial_order},{bessel_order}}}$"
        )
        projection_errors.append(
            float(abs(generated - projected) / max(abs(projected), 1.0e-14))
        )

    matrix_cache: dict[tuple[int, int, int], np.ndarray] = {}

    def finite_b_matrix(
        spherical_order: int,
        spherical_radial_order: int,
        bessel_laguerre_order: int,
    ) -> np.ndarray:
        key = (spherical_order, spherical_radial_order, bessel_laguerre_order)
        if key not in matrix_cache:
            matrix_test, matrix_field = coulomb_nonpolarized_moment_matrices(
                1,
                1,
                0.8,
                1.0,
                1.0,
                maximum_spherical_order=spherical_order,
                maximum_spherical_radial_order=spherical_radial_order,
                maximum_bessel_laguerre_order=bessel_laguerre_order,
                digits=max(32, min(digits, 40)),
            )
            matrix_cache[key] = matrix_test + matrix_field
        return matrix_cache[key]

    matrix_truncation_orders = np.asarray((2, 4, 6), dtype=int)
    matrix_truncations = [
        finite_b_matrix(8, 4, int(order)) for order in matrix_truncation_orders
    ]
    matrix_truncation_reference = matrix_truncations[-1]
    matrix_reference_norm = np.linalg.norm(matrix_truncation_reference)
    matrix_truncation_errors = np.asarray(
        [
            np.linalg.norm(matrix - matrix_truncation_reference) / matrix_reference_norm
            for matrix in matrix_truncations
        ]
    )

    spherical_truncation_cutoffs = ((3, 1), (8, 4), (9, 4))
    spherical_truncations = [
        finite_b_matrix(spherical_order, spherical_radial_order, 6)
        for spherical_order, spherical_radial_order in spherical_truncation_cutoffs
    ]
    spherical_truncation_reference = spherical_truncations[-1]
    spherical_reference_norm = np.linalg.norm(spherical_truncation_reference)
    spherical_truncation_errors = np.asarray(
        [
            np.linalg.norm(matrix - spherical_truncation_reference)
            / spherical_reference_norm
            for matrix in spherical_truncations
        ]
    )

    test_matrix, field_matrix = coulomb_nonpolarized_moment_matrices(
        3,
        1,
        0.0,
        1.0,
        1.0,
        maximum_spherical_order=5,
        maximum_spherical_radial_order=2,
        maximum_bessel_laguerre_order=0,
        digits=digits,
    )
    laguerre_sign = np.asarray(
        [(-1.0) ** laguerre for _hermite in range(4) for laguerre in range(2)]
    )
    collision_matrix = (
        laguerre_sign[:, None] * laguerre_sign[None, :] * (test_matrix + field_matrix)
    )
    symmetric_matrix = 0.5 * (collision_matrix + collision_matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric_matrix)
    published = build_collision_table(digits=digits)[2]
    published_mask = published != 0.0

    invariants = {
        "density": np.eye(8)[0],
        "parallel_momentum": np.eye(8)[2],
        "thermal_energy": np.asarray(
            [0.0, 1.0, 0.0, 0.0, 1.0 / np.sqrt(2), 0.0, 0.0, 0.0]
        ),
    }
    invariant_residuals = {
        name: float(np.linalg.norm(collision_matrix @ vector, ord=np.inf))
        for name, vector in invariants.items()
    }

    # Finite wavelength mixes particle position and gyrocenter velocity. The
    # resulting gyrocenter-density row is classical gyro-diffusion, not a
    # violation of the particle-space conservation law (Frei et al. 2021,
    # discussion following Eq. 3.5).
    gyrocenter_b = np.asarray((0.0, 0.1, 0.2, 0.4), dtype=float)
    gyrocenter_test_density_rows = []
    gyrocenter_field_density_rows = []
    gyrocenter_density_rows = []
    for b_value in gyrocenter_b:
        gyro_test, gyro_field = coulomb_nonpolarized_moment_matrices(
            1,
            1,
            float(b_value),
            1.0,
            1.0,
            maximum_spherical_order=3,
            maximum_spherical_radial_order=1,
            maximum_bessel_laguerre_order=4,
            digits=max(32, min(digits, 48)),
        )
        gyrocenter_test_density_rows.append(
            float(np.linalg.norm(gyro_test[0], ord=np.inf))
        )
        gyrocenter_field_density_rows.append(
            float(np.linalg.norm(gyro_field[0], ord=np.inf))
        )
        gyrocenter_density_rows.append(
            float(np.linalg.norm((gyro_test + gyro_field)[0], ord=np.inf))
        )
    gyrocenter_test_density_rows_array = np.asarray(gyrocenter_test_density_rows)
    gyrocenter_field_density_rows_array = np.asarray(gyrocenter_field_density_rows)
    gyrocenter_density_rows_array = np.asarray(gyrocenter_density_rows)

    def small_b_observed_order(values: np.ndarray) -> float:
        return float(np.polyfit(np.log(gyrocenter_b[1:]), np.log(values[1:]), 1)[0])

    test_observed_order = small_b_observed_order(gyrocenter_test_density_rows_array)
    field_observed_order = small_b_observed_order(gyrocenter_field_density_rows_array)
    observed_order = small_b_observed_order(gyrocenter_density_rows_array)
    metrics = {
        "final_bessel_relative_error": float(truncation_errors[-1]),
        "maximum_projection_relative_error": float(max(projection_errors)),
        "published_coefficient_maximum_absolute_error": float(
            np.max(np.abs(collision_matrix[published_mask] - published[published_mask]))
        ),
        "symmetry_maximum_absolute_error": float(
            np.max(np.abs(collision_matrix - collision_matrix.T))
        ),
        "maximum_eigenvalue": float(np.max(eigenvalues)),
        "maximum_invariant_residual": float(max(invariant_residuals.values())),
        "drift_kinetic_gyrocenter_density_row": float(gyrocenter_density_rows_array[0]),
        "finite_b_gyrocenter_density_row": float(gyrocenter_density_rows_array[-1]),
        "small_b_test_gyrocenter_diffusion_observed_order": test_observed_order,
        "small_b_field_gyrocenter_diffusion_observed_order": field_observed_order,
        "small_b_gyrocenter_diffusion_observed_order": observed_order,
        "finite_b_matrix_relative_error_at_order_4": float(
            matrix_truncation_errors[-2]
        ),
        "finite_b_spherical_relative_error_at_order_8": float(
            spherical_truncation_errors[1]
        ),
    }
    thresholds = {
        "final_bessel_relative_error": 5.0e-9,
        "maximum_projection_relative_error": 5.0e-9,
        "published_coefficient_maximum_absolute_error": 5.0e-12,
        "symmetry_maximum_absolute_error": 5.0e-12,
        "maximum_eigenvalue": 1.0e-12,
        "maximum_invariant_residual": 5.0e-12,
        "drift_kinetic_gyrocenter_density_row": 5.0e-12,
        "finite_b_gyrocenter_density_row_minimum": 1.0e-4,
        "small_b_gyrocenter_diffusion_order_minimum": 1.7,
        "small_b_gyrocenter_diffusion_order_maximum": 2.3,
        "finite_b_matrix_relative_error_at_order_4": 5.0e-6,
        "finite_b_spherical_relative_error_at_order_8": 2.0e-6,
    }
    gates = {
        name: bool(metrics[name] <= thresholds[name])
        for name in (
            "final_bessel_relative_error",
            "maximum_projection_relative_error",
            "published_coefficient_maximum_absolute_error",
            "symmetry_maximum_absolute_error",
            "maximum_eigenvalue",
            "maximum_invariant_residual",
            "drift_kinetic_gyrocenter_density_row",
            "finite_b_matrix_relative_error_at_order_4",
            "finite_b_spherical_relative_error_at_order_8",
        )
    }
    gates["finite_b_gyrocenter_diffusion_nonzero"] = bool(
        metrics["finite_b_gyrocenter_density_row"]
        >= thresholds["finite_b_gyrocenter_density_row_minimum"]
    )
    gates["small_b_gyrocenter_diffusion_quadratic"] = bool(
        thresholds["small_b_gyrocenter_diffusion_order_minimum"]
        <= observed_order
        <= thresholds["small_b_gyrocenter_diffusion_order_maximum"]
    )
    gates["small_b_test_gyrocenter_diffusion_quadratic"] = bool(
        thresholds["small_b_gyrocenter_diffusion_order_minimum"]
        <= test_observed_order
        <= thresholds["small_b_gyrocenter_diffusion_order_maximum"]
    )
    gates["small_b_field_gyrocenter_diffusion_quadratic"] = bool(
        thresholds["small_b_gyrocenter_diffusion_order_minimum"]
        <= field_observed_order
        <= thresholds["small_b_gyrocenter_diffusion_order_maximum"]
    )
    return _json_clean(
        {
            "case": "finite_b_coulomb_operator_algebra",
            "claim_scope": "offline_operator_algebra_not_runtime_transport",
            "precision_decimal_digits": int(digits),
            "basis": {"maximum_hermite_order": 3, "maximum_laguerre_order": 1},
            "truncation": {
                "coefficient": "Pi^(3,1,1)(b=1.8)",
                "orders": truncation_orders,
                "values": truncation_values,
                "relative_errors": truncation_errors,
                "reference_order": reference_order,
                "reference_value": reference_value,
            },
            "direct_projection": {
                "quadrature": "80-point Gauss-Hermite x 80-point Gauss-Laguerre",
                "labels": projection_labels,
                "relative_errors": projection_errors,
            },
            "matrix_truncation": {
                "basis": {"maximum_hermite_order": 1, "maximum_laguerre_order": 1},
                "bessel_argument": 0.8,
                "spherical_cutoff": {
                    "maximum_order": 8,
                    "maximum_radial_order": 4,
                },
                "orders": matrix_truncation_orders,
                "relative_errors": matrix_truncation_errors,
                "reference_order": int(matrix_truncation_orders[-1]),
            },
            "spherical_truncation": {
                "basis": {"maximum_hermite_order": 1, "maximum_laguerre_order": 1},
                "bessel_argument": 0.8,
                "bessel_laguerre_order": 6,
                "maximum_spherical_orders": [
                    cutoff[0] for cutoff in spherical_truncation_cutoffs
                ],
                "maximum_spherical_radial_orders": [
                    cutoff[1] for cutoff in spherical_truncation_cutoffs
                ],
                "relative_errors": spherical_truncation_errors,
                "reference_cutoff": spherical_truncation_cutoffs[-1],
            },
            "matrix": collision_matrix,
            "eigenvalues": eigenvalues,
            "invariant_residuals": invariant_residuals,
            "gyrocenter_diffusion": {
                "bessel_argument": gyrocenter_b,
                "test_density_row_infinity_norm": gyrocenter_test_density_rows_array,
                "field_density_row_infinity_norm": gyrocenter_field_density_rows_array,
                "density_row_infinity_norm": gyrocenter_density_rows_array,
                "test_small_b_observed_order": test_observed_order,
                "field_small_b_observed_order": field_observed_order,
                "small_b_observed_order": observed_order,
                "interpretation": (
                    "equation-(3.5) finite-b classical gyro-diffusion in the "
                    "test, field, and combined gyrocenter moments; particle-space "
                    "local conservation cannot be reconstructed from the "
                    "gyrophase-averaged matrix alone"
                ),
            },
            "metrics": metrics,
            "thresholds": thresholds,
            "gates": gates,
            "gate_passed": all(gates.values()),
            "references": [
                {
                    "title": "Frei et al. (2021), advanced gyrokinetic collision operators",
                    "url": "https://arxiv.org/abs/2104.11480",
                    "equations": ["3.2", "3.5", "3.41", "3.48", "3.49", "3.50"],
                },
                {
                    "title": "Abel et al. (2008), collision-operator physical constraints",
                    "url": "https://arxiv.org/abs/0808.1300",
                    "tests": ["conservation", "H-theorem", "Maxwellian null space"],
                },
            ],
        }
    )


def write_coulomb_operator_verification_figure(
    summary: dict[str, Any], out_png: Path
) -> None:
    """Write the compact paper-facing collision algebra verification panel."""

    from matplotlib.colors import TwoSlopeNorm

    colors = {
        "blue": "#16697A",
        "orange": "#E56B1F",
        "red": "#A23B3B",
        "ink": "#18232E",
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6), constrained_layout=True)
    truncation = summary["truncation"]
    errors = np.maximum(np.asarray(truncation["relative_errors"], dtype=float), 1.0e-16)
    axes[0, 0].semilogy(
        truncation["orders"],
        errors,
        "o-",
        color=colors["blue"],
        lw=2.2,
        ms=5.5,
        label=r"polarization coefficient ($N_b$)",
    )
    matrix_truncation = summary["matrix_truncation"]
    matrix_errors = np.maximum(
        np.asarray(matrix_truncation["relative_errors"], dtype=float), 1.0e-16
    )
    axes[0, 0].semilogy(
        matrix_truncation["orders"],
        matrix_errors,
        "s--",
        color=colors["orange"],
        lw=1.7,
        ms=4.8,
        label=r"assembled block ($N_b$)",
    )
    spherical_truncation = summary["spherical_truncation"]
    spherical_errors = np.maximum(
        np.asarray(spherical_truncation["relative_errors"], dtype=float), 1.0e-16
    )
    axes[0, 0].semilogy(
        spherical_truncation["maximum_spherical_orders"],
        spherical_errors,
        "D-.",
        color=colors["red"],
        lw=1.5,
        ms=4.4,
        label=r"assembled block ($p_{\max}$)",
    )
    axes[0, 0].set(
        xlabel="spectral truncation order",
        ylabel="relative truncation error",
        title="(a) Finite-$b$ spectral convergence",
    )
    axes[0, 0].legend(frameon=False, fontsize=8)

    projection = summary["direct_projection"]
    projection_errors = np.maximum(
        np.asarray(projection["relative_errors"], dtype=float), 1.0e-16
    )
    axes[0, 1].bar(
        np.arange(len(projection_errors)),
        projection_errors,
        color=colors["orange"],
        edgecolor="white",
        linewidth=0.8,
    )
    axes[0, 1].axhline(
        summary["thresholds"]["maximum_projection_relative_error"],
        color=colors["red"],
        ls="--",
        lw=1.5,
    )
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xticks(
        np.arange(len(projection_errors)), projection["labels"], fontsize=8
    )
    axes[0, 1].set(
        ylabel="relative projection error",
        title="(b) Independent velocity-space projection",
    )

    eigenvalues = np.asarray(summary["eigenvalues"], dtype=float)
    dissipation = np.maximum(-np.sort(eigenvalues), 1.0e-17)
    axes[1, 0].semilogy(
        np.arange(1, dissipation.size + 1),
        dissipation,
        "o",
        color=colors["blue"],
        ms=6,
    )
    axes[1, 0].axhspan(1.0e-17, 5.0e-13, color="#DDE9E7", alpha=0.9)
    axes[1, 0].set(
        xlabel="ordered moment-space eigenmode",
        ylabel=r"dissipation rate $-\lambda(C)$",
        title="(c) H-theorem and three invariant null modes",
    )
    axes[1, 0].text(
        0.04,
        0.06,
        "shaded: numerical null space\n"
        + rf"max invariant residual $={summary['metrics']['maximum_invariant_residual']:.1e}$",
        transform=axes[1, 0].transAxes,
        fontsize=8,
        color=colors["ink"],
    )
    gyrocenter = summary["gyrocenter_diffusion"]
    gyrocenter_b = np.asarray(gyrocenter["bessel_argument"], dtype=float)[1:]
    test_density_row = np.asarray(
        gyrocenter["test_density_row_infinity_norm"], dtype=float
    )[1:]
    field_density_row = np.asarray(
        gyrocenter["field_density_row_infinity_norm"], dtype=float
    )[1:]
    density_row = np.asarray(gyrocenter["density_row_infinity_norm"], dtype=float)[1:]
    inset = axes[1, 0].inset_axes([0.54, 0.52, 0.42, 0.40])
    inset.semilogy(
        gyrocenter_b,
        test_density_row,
        "^-.",
        color=colors["red"],
        lw=1.0,
        ms=3.2,
        label="test",
    )
    inset.semilogy(
        gyrocenter_b,
        field_density_row,
        "v:",
        color=colors["blue"],
        lw=1.0,
        ms=3.2,
        label="field",
    )
    inset.semilogy(
        gyrocenter_b,
        density_row,
        "o-",
        color=colors["orange"],
        lw=1.4,
        ms=3.8,
        label="combined",
    )
    inset.semilogy(
        gyrocenter_b,
        density_row[0] * (gyrocenter_b / gyrocenter_b[0]) ** 2,
        "--",
        color=colors["ink"],
        lw=1.0,
        label=r"$b^2$",
    )
    inset.set_title("finite-$b$ gyro-diffusion", fontsize=7)
    inset.set_xticks(gyrocenter_b, ["0.1", "0.2", "0.4"])
    inset.set_xlabel(r"$b=k_\perp\rho$", fontsize=7)
    inset.set_ylabel(r"$\|C_{0,:}\|_\infty$", fontsize=7)
    inset.tick_params(labelsize=6)
    inset.legend(frameon=False, fontsize=5.2, loc="lower right", ncol=2)
    inset.grid(alpha=0.2, lw=0.4)

    matrix = np.asarray(summary["matrix"], dtype=float)
    matrix_limit = float(np.max(np.abs(matrix)))
    image = axes[1, 1].imshow(
        matrix,
        origin="upper",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-matrix_limit, vcenter=0.0, vmax=matrix_limit),
        interpolation="nearest",
    )
    mode_labels = [rf"$({p},{j})$" for p in range(4) for j in range(2)]
    axes[1, 1].set_xticks(range(8), mode_labels, rotation=45, ha="right", fontsize=7)
    axes[1, 1].set_yticks(range(8), mode_labels, fontsize=7)
    axes[1, 1].set(
        xlabel="input moment $(p,j)$",
        ylabel="output moment $(p,j)$",
        title="(d) Drift-kinetic Coulomb moment block",
    )
    fig.colorbar(image, ax=axes[1, 1], shrink=0.82, label="normalized collision rate")

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2, lw=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Linearized Coulomb operator: algebraic and numerical closure",
        fontsize=15,
        color=colors["ink"],
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, metadata={"Software": "GKX"})
    plt.close(fig)


def write_coulomb_operator_verification_artifacts(
    out_json: Path,
    out_png: Path,
    *,
    digits: int = 80,
) -> dict[str, Any]:
    """Generate and persist the collision verification report and figure."""

    summary = build_coulomb_operator_verification_summary(digits=digits)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    write_coulomb_operator_verification_figure(summary, out_png)
    return summary


def build_drift_kinetic_response_convergence_summary(
    *,
    resolutions: tuple[tuple[int, int], ...] = (
        (3, 1),
        (5, 2),
        (7, 3),
        (9, 4),
        (11, 5),
        (15, 5),
        (20, 5),
    ),
    ion_charges: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 100.0),
    required_resolution: tuple[int, int] = (20, 5),
    nested_current_rtol: float = 5.0e-3,
    algebra_atol: float = 2.0e-12,
    original_sugama_low_charge_gap_min: float = 8.0e-2,
    original_sugama_high_charge_gap_max: float = 2.0e-2,
    improved_sugama_coulomb_gap_max: float = 1.0e-2,
    improved_sugama_correction_order: int = 5,
    paper_normalized_field: float = 1.0e-3,
    saturation_times: tuple[float, ...] = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0),
    saturation_charge: float = 1.0,
    saturation_rtol: float = 1.0e-3,
    field_linearity_rtol: float = 2.0e-12,
    spitzer_high_charge_minimum: float = 100.0,
    spitzer_high_charge_rtol: float = 8.0e-2,
    digits: int = 50,
) -> dict[str, Any]:
    r"""Build the converged driven Coulomb-response hierarchy.

    Electron--electron and electron--ion blocks use equations (3.53)--(3.56)
    of Frei et al. (2021). The electric-field source follows equation (81) of
    Frei, Ernst & Ricci (2022). The report verifies the matrix algebra and
    velocity-space convergence of the resulting current; it is deliberately
    not labelled electrical conductivity because that comparison also needs
    the paper's collision-frequency normalization and full original/improved
    Sugama hierarchies.
    """

    if not resolutions:
        raise ValueError("resolutions must contain at least one (P, J) pair")
    if any(p < 2 or j < 1 for p, j in resolutions):
        raise ValueError("response resolutions require P >= 2 and J >= 1")
    if required_resolution[0] < 2 or required_resolution[1] < 1:
        raise ValueError("required_resolution must satisfy P >= 2 and J >= 1")
    if any(not math.isfinite(z) or z <= 0.0 for z in ion_charges):
        raise ValueError("ion_charges must be finite and > 0")
    if any(right <= left for left, right in zip(ion_charges, ion_charges[1:])):
        raise ValueError("ion_charges must increase strictly")
    if nested_current_rtol <= 0.0 or not math.isfinite(nested_current_rtol):
        raise ValueError("nested_current_rtol must be finite and > 0")
    if algebra_atol <= 0.0 or not math.isfinite(algebra_atol):
        raise ValueError("algebra_atol must be finite and > 0")
    if not 0.0 < original_sugama_low_charge_gap_min < 1.0:
        raise ValueError("original_sugama_low_charge_gap_min must lie in (0, 1)")
    if not 0.0 < original_sugama_high_charge_gap_max < 1.0:
        raise ValueError("original_sugama_high_charge_gap_max must lie in (0, 1)")
    if not 0.0 < improved_sugama_coulomb_gap_max < 1.0:
        raise ValueError("improved_sugama_coulomb_gap_max must lie in (0, 1)")
    if improved_sugama_correction_order < 1:
        raise ValueError("improved_sugama_correction_order must be >= 1")
    if paper_normalized_field <= 0.0 or not math.isfinite(paper_normalized_field):
        raise ValueError("paper_normalized_field must be finite and > 0")
    if (
        not saturation_times
        or saturation_times[0] != 0.0
        or any(not math.isfinite(value) or value < 0.0 for value in saturation_times)
        or any(
            right <= left for left, right in zip(saturation_times, saturation_times[1:])
        )
    ):
        raise ValueError("saturation_times must increase strictly from zero")
    if saturation_charge not in ion_charges:
        raise ValueError("saturation_charge must be present in ion_charges")
    for name, value in (
        ("saturation_rtol", saturation_rtol),
        ("field_linearity_rtol", field_linearity_rtol),
        ("spitzer_high_charge_rtol", spitzer_high_charge_rtol),
    ):
        if value <= 0.0 or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and > 0")
    if spitzer_high_charge_minimum <= 0.0 or not math.isfinite(
        spitzer_high_charge_minimum
    ):
        raise ValueError("spitzer_high_charge_minimum must be finite and > 0")

    charge_values = np.asarray(ion_charges, dtype=float)
    solver_normalized_field = paper_normalized_field / np.sqrt(2.0)
    spitzer_conductivity = 64.0 / (3.0 * 2.0**1.5 * np.pi * charge_values)
    rows: list[dict[str, Any]] = []
    previous_current: np.ndarray | None = None
    previous_improved_current: np.ndarray | None = None
    resolution_reports: list[dict[str, Any]] = []
    saturation_report: dict[str, Any] = {}
    for maximum_hermite, maximum_laguerre in resolutions:
        maximum_degree = maximum_hermite + 2 * maximum_laguerre
        radial_limit = maximum_degree // 2
        pair_blocks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        pair_times: dict[str, float] = {}
        for label, mass_ratio in (
            ("electron_electron", 1.0),
            ("electron_ion", 1 / 1836),
        ):
            start = time.perf_counter()
            pair_blocks[label] = coulomb_drift_kinetic_moment_matrices(
                maximum_hermite,
                maximum_laguerre,
                mass_ratio,
                1.0,
                maximum_spherical_order=maximum_degree,
                maximum_spherical_radial_order=radial_limit,
                digits=digits,
            )
            pair_times[label] = time.perf_counter() - start

        n_laguerre = maximum_laguerre + 1
        mode_count = (maximum_hermite + 1) * n_laguerre
        convention_sign = np.asarray(
            [
                (-1.0) ** laguerre
                for _hermite in range(maximum_hermite + 1)
                for laguerre in range(n_laguerre)
            ]
        )

        electron_test, electron_field = pair_blocks["electron_electron"]
        ion_test, _ion_field = pair_blocks["electron_ion"]
        _original_test, original_field = original_sugama_like_species_moment_matrices(
            electron_test,
            maximum_hermite,
            maximum_laguerre,
        )
        effective_correction_order = min(
            improved_sugama_correction_order,
            maximum_laguerre,
            (maximum_hermite - 1) // 2,
        )
        _improved_test, improved_field = (
            improved_sugama_equal_temperature_moment_matrices(
                electron_test,
                maximum_hermite,
                maximum_laguerre,
                correction_order=effective_correction_order,
                digits=digits,
            )
        )
        convention = convention_sign[:, None] * convention_sign[None, :]
        electron_collision = convention * (electron_test + electron_field)
        original_electron_collision = convention * (electron_test + original_field)
        improved_electron_collision = convention * (electron_test + improved_field)
        previous_order_improved_collision: np.ndarray | None = None
        if (maximum_hermite, maximum_laguerre) == resolutions[
            -1
        ] and effective_correction_order >= 2:
            _previous_test, previous_field = (
                improved_sugama_equal_temperature_moment_matrices(
                    electron_test,
                    maximum_hermite,
                    maximum_laguerre,
                    correction_order=effective_correction_order - 1,
                    digits=digits,
                )
            )
            previous_order_improved_collision = convention * (
                electron_test + previous_field
            )
        ion_collision = convention * ion_test
        active_modes = np.asarray(
            [
                hermite * n_laguerre + laguerre
                for hermite in range(1, maximum_hermite + 1, 2)
                for laguerre in range(n_laguerre)
            ],
            dtype=int,
        )
        source = np.zeros(mode_count)
        source[n_laguerre] = -np.sqrt(2.0) * solver_normalized_field
        currents: list[float] = []
        original_currents: list[float] = []
        improved_currents: list[float] = []
        previous_order_improved_currents: list[float] = []
        solve_residuals: list[float] = []
        original_solve_residuals: list[float] = []
        improved_solve_residuals: list[float] = []
        for charge in charge_values:
            collision = electron_collision + charge * ion_collision
            original_collision = original_electron_collision + charge * ion_collision
            improved_collision = improved_electron_collision + charge * ion_collision
            previous_order_collision = (
                None
                if previous_order_improved_collision is None
                else previous_order_improved_collision + charge * ion_collision
            )
            response = np.zeros(mode_count)
            original_response = np.zeros(mode_count)
            improved_response = np.zeros(mode_count)
            response[active_modes] = np.linalg.solve(
                collision[np.ix_(active_modes, active_modes)],
                -source[active_modes],
            )
            original_response[active_modes] = np.linalg.solve(
                original_collision[np.ix_(active_modes, active_modes)],
                -source[active_modes],
            )
            improved_response[active_modes] = np.linalg.solve(
                improved_collision[np.ix_(active_modes, active_modes)],
                -source[active_modes],
            )
            currents.append(float(abs(response[n_laguerre] / np.sqrt(2.0))))
            original_currents.append(
                float(abs(original_response[n_laguerre] / np.sqrt(2.0)))
            )
            improved_currents.append(
                float(abs(improved_response[n_laguerre] / np.sqrt(2.0)))
            )
            if previous_order_collision is not None:
                previous_order_response = np.zeros(mode_count)
                previous_order_response[active_modes] = np.linalg.solve(
                    previous_order_collision[np.ix_(active_modes, active_modes)],
                    -source[active_modes],
                )
                previous_order_improved_currents.append(
                    float(abs(previous_order_response[n_laguerre] / np.sqrt(2.0)))
                )
            solve_residuals.append(
                float(np.max(np.abs((collision @ response + source)[active_modes])))
            )
            original_solve_residuals.append(
                float(
                    np.max(
                        np.abs(
                            (original_collision @ original_response + source)[
                                active_modes
                            ]
                        )
                    )
                )
            )
            improved_solve_residuals.append(
                float(
                    np.max(
                        np.abs(
                            (improved_collision @ improved_response + source)[
                                active_modes
                            ]
                        )
                    )
                )
            )

        current_array = np.asarray(currents)
        original_current_array = np.asarray(original_currents)
        improved_current_array = np.asarray(improved_currents)
        correction_order_relative_change = (
            None
            if not previous_order_improved_currents
            else np.abs(
                improved_current_array - np.asarray(previous_order_improved_currents)
            )
            / np.maximum(np.abs(improved_current_array), np.finfo(float).tiny)
        )
        relative_change = (
            None
            if previous_current is None
            else np.abs(current_array - previous_current)
            / np.maximum(np.abs(current_array), np.finfo(float).tiny)
        )
        previous_current = current_array
        improved_relative_change = (
            None
            if previous_improved_current is None
            else np.abs(improved_current_array - previous_improved_current)
            / np.maximum(np.abs(improved_current_array), np.finfo(float).tiny)
        )
        previous_improved_current = improved_current_array

        density = np.zeros(mode_count)
        density[0] = 1.0
        momentum = np.zeros(mode_count)
        momentum[n_laguerre] = 1.0
        energy = np.zeros(mode_count)
        if maximum_laguerre >= 1 and maximum_hermite >= 2:
            energy[1] = 1.0
            energy[2 * n_laguerre] = 1 / np.sqrt(2.0)
        invariant_residuals = {
            name: float(np.max(np.abs(electron_collision @ vector)))
            for name, vector in (
                ("density", density),
                ("parallel_momentum", momentum),
                ("thermal_energy", energy),
            )
        }
        original_invariant_residuals = {
            name: float(np.max(np.abs(original_electron_collision @ vector)))
            for name, vector in (
                ("density", density),
                ("parallel_momentum", momentum),
                ("thermal_energy", energy),
            )
        }
        improved_invariant_residuals = {
            name: float(np.max(np.abs(improved_electron_collision @ vector)))
            for name, vector in (
                ("density", density),
                ("parallel_momentum", momentum),
                ("thermal_energy", energy),
            )
        }
        collision_models = (
            electron_collision,
            original_electron_collision,
            improved_electron_collision,
        )
        symmetry_error = max(
            float(np.max(np.abs(matrix - matrix.T))) for matrix in collision_models
        )
        maximum_eigenvalue = max(
            float(np.linalg.eigvalsh(0.5 * (matrix + matrix.T)).max())
            for matrix in collision_models
        )
        maximum_change = (
            None if relative_change is None else float(np.max(relative_change))
        )
        if (maximum_hermite, maximum_laguerre) == resolutions[-1]:
            charge = float(saturation_charge)
            time_values = np.asarray(saturation_times, dtype=float)
            paper_field_values = paper_normalized_field * np.asarray(
                (0.1, 1.0, 10.0), dtype=float
            )
            current_mode = int(np.flatnonzero(active_modes == n_laguerre)[0])
            model_matrices = {
                "coulomb": electron_collision + charge * ion_collision,
                "original_sugama": (
                    original_electron_collision + charge * ion_collision
                ),
                "improved_sugama": (
                    improved_electron_collision + charge * ion_collision
                ),
            }
            saturation_models: dict[str, Any] = {}
            for model, collision_matrix in model_matrices.items():
                reduced = collision_matrix[np.ix_(active_modes, active_modes)]
                reduced = 0.5 * (reduced + reduced.T)
                steady = np.linalg.solve(reduced, -source[active_modes])
                eigenvalues, eigenvectors = np.linalg.eigh(reduced)
                transient_coefficients = eigenvectors.T @ steady
                transient = (
                    np.exp(time_values[:, None] * eigenvalues[None, :])
                    * transient_coefficients[None, :]
                ) @ eigenvectors.T
                trajectory = steady[None, :] - transient
                current_trace = trajectory[:, current_mode] / np.sqrt(2.0)
                stationary_current = float(steady[current_mode] / np.sqrt(2.0))

                conductivity_values = []
                for paper_field in paper_field_values:
                    solver_field = paper_field / np.sqrt(2.0)
                    drive = np.zeros_like(source[active_modes])
                    drive[current_mode] = -np.sqrt(2.0) * solver_field
                    field_response = np.linalg.solve(reduced, -drive)
                    conductivity_values.append(
                        float(
                            abs(field_response[current_mode] / np.sqrt(2.0))
                            / solver_field
                        )
                    )
                conductivity_array = np.asarray(conductivity_values)
                reference_conductivity = conductivity_array[1]
                saturation_models[model] = {
                    "current_over_vte": current_trace.tolist(),
                    "stationary_current_over_vte": stationary_current,
                    "saturation_relative_error": float(
                        abs(current_trace[-1] - stationary_current)
                        / max(abs(stationary_current), np.finfo(float).tiny)
                    ),
                    "conductivity_over_ne2_mnu": conductivity_array.tolist(),
                    "field_linearity_relative_error": float(
                        np.max(
                            np.abs(conductivity_array / reference_conductivity - 1.0)
                        )
                    ),
                }
            saturation_report = {
                "ion_charge": charge,
                "time_nu_ee": time_values.tolist(),
                "paper_normalized_field": paper_normalized_field,
                "solver_normalized_field": solver_normalized_field,
                "paper_field_scan": paper_field_values.tolist(),
                "models": saturation_models,
                "maximum_saturation_relative_error": max(
                    row["saturation_relative_error"]
                    for row in saturation_models.values()
                ),
                "maximum_field_linearity_relative_error": max(
                    row["field_linearity_relative_error"]
                    for row in saturation_models.values()
                ),
            }
        report = {
            "maximum_hermite_order": maximum_hermite,
            "maximum_laguerre_order": maximum_laguerre,
            "mode_count": mode_count,
            "maximum_spherical_order": maximum_degree,
            "maximum_spherical_radial_order": radial_limit,
            "improved_sugama_correction_order": effective_correction_order,
            "current": current_array.tolist(),
            "original_sugama_current": original_current_array.tolist(),
            "original_sugama_relative_gap": (
                (current_array - original_current_array)
                / np.maximum(np.abs(current_array), np.finfo(float).tiny)
            ).tolist(),
            "improved_sugama_current": improved_current_array.tolist(),
            "improved_sugama_relative_gap": (
                (improved_current_array - current_array)
                / np.maximum(np.abs(current_array), np.finfo(float).tiny)
            ).tolist(),
            "conductivity_over_ne2_mnu": (
                current_array / solver_normalized_field
            ).tolist(),
            "original_sugama_conductivity_over_ne2_mnu": (
                original_current_array / solver_normalized_field
            ).tolist(),
            "improved_sugama_conductivity_over_ne2_mnu": (
                improved_current_array / solver_normalized_field
            ).tolist(),
            "spitzer_relative_error": (
                np.abs(current_array / solver_normalized_field - spitzer_conductivity)
                / spitzer_conductivity
            ).tolist(),
            "relative_change_from_previous": (
                None if relative_change is None else relative_change.tolist()
            ),
            "maximum_relative_change": maximum_change,
            "improved_relative_change_from_previous": (
                None
                if improved_relative_change is None
                else improved_relative_change.tolist()
            ),
            "improved_maximum_relative_change": (
                None
                if improved_relative_change is None
                else float(np.max(improved_relative_change))
            ),
            "improved_correction_order_relative_change": (
                None
                if correction_order_relative_change is None
                else correction_order_relative_change.tolist()
            ),
            "improved_correction_order_maximum_change": (
                None
                if correction_order_relative_change is None
                else float(np.max(correction_order_relative_change))
            ),
            "invariant_residuals": invariant_residuals,
            "original_sugama_invariant_residuals": original_invariant_residuals,
            "improved_sugama_invariant_residuals": improved_invariant_residuals,
            "symmetry_max_abs": symmetry_error,
            "maximum_eigenvalue": maximum_eigenvalue,
            "solve_residual_max": float(
                max(
                    (
                        *solve_residuals,
                        *original_solve_residuals,
                        *improved_solve_residuals,
                    )
                )
            ),
            "generation_seconds": pair_times,
        }
        resolution_reports.append(report)
        for charge_index, charge in enumerate(charge_values):
            rows.append(
                {
                    "maximum_hermite_order": maximum_hermite,
                    "maximum_laguerre_order": maximum_laguerre,
                    "mode_count": mode_count,
                    "ion_charge": float(charge),
                    "normalized_current": float(current_array[charge_index]),
                    "original_sugama_normalized_current": float(
                        original_current_array[charge_index]
                    ),
                    "improved_sugama_normalized_current": float(
                        improved_current_array[charge_index]
                    ),
                    "conductivity_over_ne2_mnu": float(
                        current_array[charge_index] / solver_normalized_field
                    ),
                    "spitzer_conductivity_over_ne2_mnu": float(
                        spitzer_conductivity[charge_index]
                    ),
                    "relative_change_from_previous": (
                        None
                        if relative_change is None
                        else float(relative_change[charge_index])
                    ),
                }
            )

    final = resolution_reports[-1]
    resolution_reached = (
        final["maximum_hermite_order"] >= required_resolution[0]
        and final["maximum_laguerre_order"] >= required_resolution[1]
    )
    high_charge_index = int(np.argmax(charge_values))
    gates = {
        "required_resolution_reached": resolution_reached,
        "nested_current_converged": (
            final["maximum_relative_change"] is not None
            and final["maximum_relative_change"] <= nested_current_rtol
        ),
        "collision_invariants": max(final["invariant_residuals"].values())
        <= algebra_atol,
        "original_sugama_invariants": max(
            final["original_sugama_invariant_residuals"].values()
        )
        <= algebra_atol,
        "improved_sugama_invariants": max(
            final["improved_sugama_invariant_residuals"].values()
        )
        <= algebra_atol,
        "self_adjoint_symmetry": final["symmetry_max_abs"] <= algebra_atol,
        "nonpositive_spectrum": final["maximum_eigenvalue"] <= algebra_atol,
        "driven_solve_residual": final["solve_residual_max"] <= algebra_atol,
        "original_sugama_low_charge_underprediction": final[
            "original_sugama_relative_gap"
        ][0]
        >= original_sugama_low_charge_gap_min,
        "original_sugama_high_charge_convergence": final[
            "original_sugama_relative_gap"
        ][-1]
        <= original_sugama_high_charge_gap_max,
        "improved_sugama_order_reached": final["improved_sugama_correction_order"]
        >= improved_sugama_correction_order,
        "improved_sugama_nested_current_converged": final[
            "improved_maximum_relative_change"
        ]
        is not None
        and final["improved_maximum_relative_change"] <= nested_current_rtol,
        "improved_sugama_correction_hierarchy_converged": final[
            "improved_correction_order_maximum_change"
        ]
        is not None
        and final["improved_correction_order_maximum_change"] <= nested_current_rtol,
        "improved_sugama_matches_coulomb": max(
            abs(value) for value in final["improved_sugama_relative_gap"]
        )
        <= improved_sugama_coulomb_gap_max,
        "spitzer_high_charge_asymptote": (
            charge_values[high_charge_index] >= spitzer_high_charge_minimum
            and final["spitzer_relative_error"][high_charge_index]
            <= spitzer_high_charge_rtol
        ),
        "stationary_current_saturated": saturation_report[
            "maximum_saturation_relative_error"
        ]
        <= saturation_rtol,
        "electric_field_linearity": saturation_report[
            "maximum_field_linearity_relative_error"
        ]
        <= field_linearity_rtol,
    }
    return {
        "schema_version": 4,
        "title": "Drift-kinetic Coulomb and Sugama conductivity convergence",
        "literature_equations": {
            "collision": "Frei et al. (2021), equations (3.53)--(3.56)",
            "original_sugama": "Sugama, Watanabe & Nunami (2009)",
            "improved_sugama": (
                "Frei, Ernst & Ricci (2022), equations (45), (60), (79)--(81)"
            ),
            "drive": "Frei, Ernst & Ricci (2022), equation (81)",
            "conductivity": "Frei, Ernst & Ricci (2022), equations (83)--(84) and Figures 15--16",
        },
        "scope": (
            "Converged Coulomb and arbitrary-order, equal-temperature original- and "
            "improved-Sugama conductivity, including dimensional normalization, "
            "electric-field linearity, stationary saturation, and the high-charge "
            "Spitzer asymptote."
        ),
        "ion_charge": charge_values.tolist(),
        "normalized_field": solver_normalized_field,
        "field_normalization": {
            "paper": "e E / (sqrt(m_e T_e) nu_ee)",
            "paper_value": paper_normalized_field,
            "solver": "e E / (m_e v_Te nu_ee)",
            "solver_value": solver_normalized_field,
            "thermal_speed": "v_Te = sqrt(2 T_e / m_e)",
        },
        "conductivity_normalization": {
            "computed": "sigma_parallel / (n_e e^2 / (m_e nu_ee))",
            "computed_value": (
                np.asarray(final["conductivity_over_ne2_mnu"], dtype=float)
            ).tolist(),
            "spitzer_high_charge": ("64 / (3 * 2^(3/2) * pi * Z)"),
            "spitzer_value": spitzer_conductivity.tolist(),
            "high_charge_relative_error": final["spitzer_relative_error"][
                high_charge_index
            ],
        },
        "saturation": saturation_report,
        "required_resolution": list(required_resolution),
        "thresholds": {
            "nested_current_rtol": nested_current_rtol,
            "algebra_atol": algebra_atol,
            "original_sugama_low_charge_gap_min": (original_sugama_low_charge_gap_min),
            "original_sugama_high_charge_gap_max": (
                original_sugama_high_charge_gap_max
            ),
            "improved_sugama_coulomb_gap_max": improved_sugama_coulomb_gap_max,
            "improved_sugama_correction_order": improved_sugama_correction_order,
            "saturation_rtol": saturation_rtol,
            "field_linearity_rtol": field_linearity_rtol,
            "spitzer_high_charge_minimum": spitzer_high_charge_minimum,
            "spitzer_high_charge_rtol": spitzer_high_charge_rtol,
        },
        "resolutions": resolution_reports,
        "rows": rows,
        "gates": gates,
        "gate_passed": all(gates.values()),
    }


def write_drift_kinetic_response_convergence_figure(
    summary: dict[str, Any], out_png: Path
) -> None:
    """Write the publication panel for the driven-response hierarchy."""

    colors = {
        "navy": "#17324D",
        "blue": "#2878A8",
        "teal": "#16817A",
        "orange": "#D97732",
        "red": "#B33A3A",
        "ink": "#20272E",
    }
    reports = summary["resolutions"]
    final_report = reports[-1]
    charges = np.asarray(summary["ion_charge"], dtype=float)
    mode_counts = np.asarray([row["mode_count"] for row in reports], dtype=float)
    labels = [
        f"({row['maximum_hermite_order']}, {row['maximum_laguerre_order']})"
        for row in reports
    ]
    palette = plt.get_cmap("viridis")(np.linspace(0.15, 0.88, len(reports)))
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0), constrained_layout=True)

    for color, label, row in zip(palette, labels, reports, strict=True):
        axes[0, 0].semilogx(
            charges,
            np.asarray(row["current"], dtype=float) / summary["normalized_field"],
            "o-",
            color=color,
            lw=1.7,
            ms=4.2,
            label=label,
        )
    axes[0, 0].semilogx(
        charges,
        np.asarray(final_report["original_sugama_current"], dtype=float)
        / summary["normalized_field"],
        "^--",
        color=colors["ink"],
        markerfacecolor="white",
        lw=1.8,
        ms=5.0,
        label=(
            "original Sugama "
            f"({final_report['maximum_hermite_order']}, "
            f"{final_report['maximum_laguerre_order']})"
        ),
    )
    axes[0, 0].semilogx(
        charges,
        np.asarray(final_report["improved_sugama_current"], dtype=float)
        / summary["normalized_field"],
        "x-.",
        color=colors["red"],
        lw=1.8,
        ms=5.0,
        markeredgewidth=1.5,
        label=(
            "improved Sugama "
            f"({final_report['maximum_hermite_order']}, "
            f"{final_report['maximum_laguerre_order']}; "
            f"K={final_report['improved_sugama_correction_order']})"
        ),
    )
    axes[0, 0].semilogx(
        charges,
        np.asarray(summary["conductivity_normalization"]["spitzer_value"], dtype=float),
        "o:",
        color=colors["ink"],
        markerfacecolor="white",
        lw=1.5,
        ms=4.4,
        label="Spitzer high-$Z$ asymptote",
    )
    axes[0, 0].set(
        xlabel="ion charge $Z$",
        ylabel=r"$\sigma_\parallel/[n_e e^2/(m_e\nu_{ee})]$",
        title="(a) Conductivity hierarchy",
    )
    axes[0, 0].legend(
        title=r"$(P,J)$", frameon=False, fontsize=6.8, title_fontsize=8, ncol=3
    )

    for charge_index, charge in enumerate(charges):
        x_values: list[float] = []
        y_values: list[float] = []
        for row in reports[1:]:
            change = row["relative_change_from_previous"]
            if change is not None:
                x_values.append(float(row["mode_count"]))
                y_values.append(max(float(change[charge_index]), 1.0e-16))
        axes[0, 1].semilogy(
            x_values,
            y_values,
            "o-",
            lw=1.5,
            ms=4.0,
            label=rf"$Z={charge:g}$",
        )
    axes[0, 1].axhline(
        summary["thresholds"]["nested_current_rtol"],
        color=colors["red"],
        ls="--",
        lw=1.4,
        label="acceptance",
    )
    axes[0, 1].set(
        xlabel="retained Hermite--Laguerre modes",
        ylabel="nested relative current change",
        title="(b) Velocity-space convergence",
    )
    axes[0, 1].legend(frameon=False, fontsize=7, ncol=2)

    invariant = np.asarray(
        [
            max(
                *row["invariant_residuals"].values(),
                *row["original_sugama_invariant_residuals"].values(),
                *row["improved_sugama_invariant_residuals"].values(),
            )
            for row in reports
        ]
    )
    symmetry = np.asarray([row["symmetry_max_abs"] for row in reports])
    positive_eigenvalue = np.maximum(
        np.asarray([row["maximum_eigenvalue"] for row in reports]), 0.0
    )
    solve_residual = np.asarray([row["solve_residual_max"] for row in reports])
    for values, marker, label, color in (
        (invariant, "o", "invariants", colors["blue"]),
        (symmetry, "s", "self-adjoint symmetry", colors["orange"]),
        (positive_eigenvalue, "^", "positive spectrum", colors["red"]),
        (solve_residual, "D", "forced solve", colors["teal"]),
    ):
        axes[1, 0].semilogy(
            mode_counts,
            np.maximum(values, 1.0e-18),
            marker + "-",
            lw=1.5,
            ms=4.2,
            color=color,
            label=label,
        )
    axes[1, 0].axhline(
        summary["thresholds"]["algebra_atol"],
        color=colors["ink"],
        ls="--",
        lw=1.2,
        label="acceptance",
    )
    axes[1, 0].set(
        xlabel="retained Hermite--Laguerre modes",
        ylabel="absolute residual",
        title="(c) Conservation and dissipation gates",
    )
    axes[1, 0].legend(frameon=False, fontsize=7, ncol=2)

    saturation = summary["saturation"]
    time_values = np.asarray(saturation["time_nu_ee"], dtype=float)
    for model, label, style, color in (
        ("coulomb", "Coulomb", "-", colors["blue"]),
        ("original_sugama", "original Sugama", "--", colors["ink"]),
        ("improved_sugama", "improved Sugama $K=5$", "-.", colors["red"]),
    ):
        axes[1, 1].plot(
            time_values,
            1.0e3 * np.asarray(saturation["models"][model]["current_over_vte"]),
            style,
            color=color,
            lw=1.8,
            label=label,
        )
    axes[1, 1].set(
        xlabel=r"$t\nu_{ee}$",
        ylabel=r"$10^3 u_e/v_{Te}$",
        title="(d) Stationary-current saturation ($Z=1$)",
    )
    axes[1, 1].legend(frameon=False, fontsize=8)

    for axis in axes.flat:
        axis.grid(alpha=0.22, lw=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    status = "PASS" if summary["gate_passed"] else "OPEN"
    fig.suptitle(
        f"Drift-kinetic Coulomb/Sugama conductivity: resolved hierarchies ({status})",
        fontsize=15,
        color=colors["ink"],
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, metadata={"Software": "GKX"})
    fig.savefig(out_png.with_suffix(".pdf"), metadata={"Creator": "GKX"})
    plt.close(fig)


def write_drift_kinetic_response_convergence_artifacts(
    out_json: Path,
    out_csv: Path,
    out_png: Path,
    **summary_kwargs: Any,
) -> dict[str, Any]:
    """Generate compact machine-readable and publication response artifacts."""

    summary = build_drift_kinetic_response_convergence_summary(**summary_kwargs)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary["rows"]).to_csv(out_csv, index=False)
    write_drift_kinetic_response_convergence_figure(summary, out_png)
    return summary


def finite_wavelength_itg_growth_curve(
    table_path: Path,
    collision_frequencies: np.ndarray,
) -> dict[str, Any]:
    r"""Evaluate the dominant slab-ITG eigenvalue from one exact pair table.

    The protocol follows Frei, Hoffmann & Ricci (2022), Figure 16: slab
    geometry, :math:`k_\perp=0.5`, :math:`k_\parallel=0.1`,
    :math:`\eta=3`, and :math:`\tau=1`. The table archive stores the paper's
    Hermite-major, unsigned-Laguerre convention; this routine converts it to
    the runtime convention before probing the complete solved-field RHS.
    """

    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from gkx.config import GridConfig
    from gkx.core.grid import build_spectral_grid
    from gkx.geometry import SlabGeometry
    from gkx.operators.linear.cache_builder import build_linear_cache
    from gkx.operators.linear.params import LinearParams, LinearTerms
    from gkx.operators.linear.rhs import linear_rhs_cached
    from gkx.operators.linear.collisions import (
        FiniteWavelengthCoulombOperator,
    )

    frequencies = np.asarray(collision_frequencies, dtype=float)
    if frequencies.ndim != 1 or frequencies.size < 2:
        raise ValueError("collision_frequencies must contain at least two values")
    if np.any(~np.isfinite(frequencies)) or np.any(frequencies < 0.0):
        raise ValueError("collision_frequencies must be finite and >= 0")
    if np.any(np.diff(frequencies) <= 0.0):
        raise ValueError("collision_frequencies must be strictly increasing")

    with np.load(table_path) as archive:
        resolution = np.asarray(archive["resolution"], dtype=int)
        if resolution.shape != (2,):
            raise ValueError(
                "table resolution must contain Hermite and Laguerre maxima"
            )
        maximum_hermite, maximum_laguerre = map(int, resolution)
        bessel_argument = float(archive["bessel_argument"])
        paper_kperp = float(archive["paper_kperp"])
        raw = tuple(np.asarray(archive[f"array_{index}"]) for index in range(6))

    nm, nl = maximum_hermite + 1, maximum_laguerre + 1
    mode_count = nm * nl
    expected_shapes = ((mode_count, mode_count),) * 2 + ((mode_count,),) * 4
    if tuple(array.shape for array in raw) != expected_shapes:
        raise ValueError("table arrays do not match the declared velocity resolution")
    laguerre_sign = np.asarray(
        [(-1.0) ** order for _p in range(nm) for order in range(nl)]
    )
    runtime_arrays: list[jnp.ndarray] = []
    for index, array in enumerate(raw):
        converted = (
            laguerre_sign[:, None] * laguerre_sign[None, :] * array
            if index < 2
            else laguerre_sign * array
        )
        table = np.zeros((1, 1, 2, 2) + converted.shape)
        table[0, 0, 1, 1] = converted
        runtime_arrays.append(jnp.asarray(table))

    grid = build_spectral_grid(
        GridConfig(Nx=1, Ny=4, Nz=8, Lx=2.0 * np.pi, Ly=4.0 * np.pi)
    )
    parameters = LinearParams(
        charge_sign=jnp.asarray([1.0]),
        density=jnp.asarray([1.0]),
        mass=jnp.asarray([1.0]),
        temp=jnp.asarray([1.0]),
        tau_e=1.0,
        vth=jnp.asarray([1.0]),
        rho=jnp.asarray([1.0]),
        tz=jnp.asarray([1.0]),
        kpar_scale=0.1,
        fprim=jnp.asarray([1.0]),
        tprim=jnp.asarray([3.0]),
    )
    cache = build_linear_cache(
        grid,
        SlabGeometry(s_hat=0.0, z0=10.0),
        parameters,
        Nl=nl,
        Nm=nm,
    )
    terms = LinearTerms(
        streaming=1.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=1.0,
        collisions=1.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )
    phase = np.exp(1j * np.asarray(grid.z))

    def rhs_matrix(collision_frequency: float) -> np.ndarray:
        operator = FiniteWavelengthCoulombOperator(
            jnp.asarray([0.0, bessel_argument]),
            jnp.asarray([[collision_frequency]]),
            *runtime_arrays,
        )
        matrix = np.empty((mode_count, mode_count), dtype=complex)
        for column in range(mode_count):
            laguerre_order, hermite_order = divmod(column, nm)
            state = np.zeros((1, nl, nm, 4, 1, 8), dtype=complex)
            state[0, laguerre_order, hermite_order, 1, 0] = phase
            rhs, _ = linear_rhs_cached(
                jnp.asarray(state),
                cache,
                parameters,
                terms=terms,
                collision_operator=operator,
                use_jit=False,
            )
            matrix[:, column] = np.asarray(rhs)[0, :, :, 1, 0, 0].reshape(-1) / phase[0]
        return matrix

    collisionless = rhs_matrix(0.0)
    collision_matrix = rhs_matrix(1.0) - collisionless
    growth: list[float] = []
    frequency: list[float] = []
    for collision_frequency in frequencies:
        eigenvalues = np.linalg.eigvals(
            collisionless + collision_frequency * collision_matrix
        )
        dominant = eigenvalues[int(np.argmax(eigenvalues.real))]
        growth.append(float(dominant.real))
        frequency.append(float(dominant.imag))
    jax.clear_caches()
    return {
        "maximum_hermite_order": maximum_hermite,
        "maximum_laguerre_order": maximum_laguerre,
        "mode_count": mode_count,
        "bessel_argument": bessel_argument,
        "paper_kperp": paper_kperp,
        "collision_frequency": frequencies.tolist(),
        "growth": growth,
        "frequency": frequency,
    }


def collisionless_slab_itg_hierarchy(
    resolutions: tuple[tuple[int, int], ...],
) -> list[dict[str, float | int]]:
    """Evaluate the paper-protocol collisionless endpoint versus resolution."""

    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from gkx.config import GridConfig
    from gkx.core.grid import build_spectral_grid
    from gkx.geometry import SlabGeometry
    from gkx.operators.linear.cache_builder import build_linear_cache
    from gkx.operators.linear.params import LinearParams, LinearTerms
    from gkx.operators.linear.rhs import linear_rhs_cached

    if len(resolutions) < 2 or any(p < 0 or j < 0 for p, j in resolutions):
        raise ValueError("at least two non-negative resolutions are required")
    grid = build_spectral_grid(
        GridConfig(Nx=1, Ny=4, Nz=8, Lx=2.0 * np.pi, Ly=4.0 * np.pi)
    )
    parameters = LinearParams(
        charge_sign=jnp.asarray([1.0]),
        density=jnp.asarray([1.0]),
        mass=jnp.asarray([1.0]),
        temp=jnp.asarray([1.0]),
        tau_e=1.0,
        vth=jnp.asarray([1.0]),
        rho=jnp.asarray([1.0]),
        tz=jnp.asarray([1.0]),
        kpar_scale=0.1,
        fprim=jnp.asarray([1.0]),
        tprim=jnp.asarray([3.0]),
    )
    terms = LinearTerms(
        streaming=1.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=1.0,
        collisions=0.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )
    phase = np.exp(1j * np.asarray(grid.z))
    rows: list[dict[str, float | int]] = []
    for maximum_hermite, maximum_laguerre in resolutions:
        nm, nl = maximum_hermite + 1, maximum_laguerre + 1
        mode_count = nm * nl
        cache = build_linear_cache(
            grid,
            SlabGeometry(s_hat=0.0, z0=10.0),
            parameters,
            Nl=nl,
            Nm=nm,
        )
        matrix = np.empty((mode_count, mode_count), dtype=complex)
        for column in range(mode_count):
            laguerre_order, hermite_order = divmod(column, nm)
            state = np.zeros((1, nl, nm, 4, 1, 8), dtype=complex)
            state[0, laguerre_order, hermite_order, 1, 0] = phase
            rhs, _ = linear_rhs_cached(
                jnp.asarray(state), cache, parameters, terms=terms, use_jit=False
            )
            matrix[:, column] = np.asarray(rhs)[0, :, :, 1, 0, 0].reshape(-1) / phase[0]
        eigenvalues = np.linalg.eigvals(matrix)
        dominant = eigenvalues[int(np.argmax(eigenvalues.real))]
        rows.append(
            {
                "maximum_hermite_order": maximum_hermite,
                "maximum_laguerre_order": maximum_laguerre,
                "mode_count": mode_count,
                "growth": float(dominant.real),
                "frequency": float(dominant.imag),
            }
        )
    jax.clear_caches()
    return rows


def summarize_finite_wavelength_itg_curves(
    curves: list[dict[str, Any]],
    *,
    collisionless_hierarchy: list[dict[str, Any]] | None = None,
    convergence_rtol: float = 0.05,
    collisionless_endpoint_rtol: float = 0.01,
    resolved_collision_minimum: float = 0.03,
) -> dict[str, Any]:
    """Build fail-closed growth convergence metrics from ordered resolutions."""

    if len(curves) < 2:
        raise ValueError("at least two resolution curves are required")
    ordered = sorted(
        curves,
        key=lambda row: (
            int(row["maximum_hermite_order"]),
            int(row["maximum_laguerre_order"]),
        ),
    )
    collision_frequency = np.asarray(ordered[0]["collision_frequency"], dtype=float)
    if any(
        not np.array_equal(
            collision_frequency, np.asarray(row["collision_frequency"], dtype=float)
        )
        for row in ordered[1:]
    ):
        raise ValueError("all curves must use the same collision-frequency grid")
    comparisons: list[dict[str, Any]] = []
    for lower, upper in zip(ordered[:-1], ordered[1:], strict=True):
        lower_growth = np.asarray(lower["growth"], dtype=float)
        upper_growth = np.asarray(upper["growth"], dtype=float)
        scale = np.maximum(np.abs(upper_growth), 1.0e-12)
        relative_change = np.abs(lower_growth - upper_growth) / scale
        resolved = collision_frequency >= resolved_collision_minimum
        unstable = upper_growth > 0.0
        comparisons.append(
            {
                "lower_resolution": [
                    int(lower["maximum_hermite_order"]),
                    int(lower["maximum_laguerre_order"]),
                ],
                "upper_resolution": [
                    int(upper["maximum_hermite_order"]),
                    int(upper["maximum_laguerre_order"]),
                ],
                "relative_growth_change": relative_change.tolist(),
                "maximum_all_frequency_relative_change": float(np.max(relative_change)),
                "maximum_resolved_unstable_relative_change": float(
                    np.max(relative_change[resolved & unstable])
                ),
            }
        )
    final_change = comparisons[-1]
    low_collisionality_converged = (
        final_change["maximum_all_frequency_relative_change"] <= convergence_rtol
    )
    resolved_unstable_converged = (
        final_change["maximum_resolved_unstable_relative_change"] <= convergence_rtol
    )
    endpoint_change = None
    if collisionless_hierarchy is not None:
        if len(collisionless_hierarchy) < 2:
            raise ValueError("collisionless_hierarchy must contain at least two rows")
        endpoint_growth = np.asarray(
            [row["growth"] for row in collisionless_hierarchy[-2:]], dtype=float
        )
        endpoint_change = float(
            abs(endpoint_growth[1] - endpoint_growth[0])
            / max(abs(endpoint_growth[1]), 1.0e-12)
        )
    gates = {
        "paper_wavelength_reproduced": bool(
            np.isclose(float(ordered[-1]["bessel_argument"]), 1.0 / np.sqrt(2.0))
        ),
        "collisionless_p15_p18_converged": (
            endpoint_change is not None
            and endpoint_change <= collisionless_endpoint_rtol
        ),
        "intermediate_collision_range_converged": resolved_unstable_converged,
        "low_collisionality_growth_converged": low_collisionality_converged,
        "literature_resolution_reached": (
            int(ordered[-1]["maximum_hermite_order"]) >= 18
            and int(ordered[-1]["maximum_laguerre_order"]) >= 6
        ),
    }
    gates["equivalent_growth_convergence_reached"] = bool(
        int(ordered[-1]["maximum_hermite_order"]) >= 15
        and int(ordered[-1]["maximum_laguerre_order"]) >= 6
        and gates["collisionless_p15_p18_converged"]
        and resolved_unstable_converged
        and low_collisionality_converged
    )
    gate_passed = bool(
        gates["paper_wavelength_reproduced"]
        and gates["collisionless_p15_p18_converged"]
        and gates["intermediate_collision_range_converged"]
        and gates["low_collisionality_growth_converged"]
        and (
            gates["literature_resolution_reached"]
            or gates["equivalent_growth_convergence_reached"]
        )
    )
    return {
        "schema_version": 2,
        "claim_scope": "paper_protocol_slab_itg_equivalent_growth_convergence",
        "literature_reference": (
            "Frei, Hoffmann & Ricci (2022), equations (2.14)--(2.18) and Figure 16"
        ),
        "protocol": {
            "geometry": "homogeneous slab",
            "paper_kperp": 0.5,
            "kparallel": 0.1,
            "eta": 3.0,
            "tau": 1.0,
            "bessel_argument": 1.0 / np.sqrt(2.0),
            "resolved_collision_minimum": resolved_collision_minimum,
        },
        "thresholds": {
            "nested_growth_relative_tolerance": convergence_rtol,
            "collisionless_endpoint_relative_tolerance": collisionless_endpoint_rtol,
        },
        "literature_required_resolution": [18, 6],
        "curves": ordered,
        "collisionless_hierarchy": collisionless_hierarchy,
        "collisionless_endpoint_relative_change": endpoint_change,
        "comparisons": comparisons,
        "gates": gates,
        "gate_passed": gate_passed,
    }


def write_finite_wavelength_itg_figure(summary: dict[str, Any], out_png: Path) -> None:
    """Write the paper-protocol growth and nested-convergence panel."""

    colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.88, len(summary["curves"])))
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 7.2), constrained_layout=True)
    fig.patch.set_facecolor("white")
    for color, curve in zip(colors, summary["curves"], strict=True):
        nu = np.asarray(curve["collision_frequency"], dtype=float)
        axes[0].plot(
            nu,
            curve["growth"],
            "o-",
            color=color,
            lw=1.8,
            ms=3.7,
            label=(
                rf"$(P,J)=({curve['maximum_hermite_order']},"
                rf"{curve['maximum_laguerre_order']})$"
            ),
        )
    axes[0].axhline(0.0, color="#20272E", lw=0.9)
    axes[0].set_xscale("symlog", linthresh=1.0e-4)
    axes[0].set_xlim(0.0, 11.0)
    axes[0].set(
        xlabel=r"collision frequency $\nu$",
        ylabel=r"dominant growth rate $\gamma$",
        title="(a) Slab ITG collisional stabilization",
    )
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    collisionless = summary.get("collisionless_hierarchy")
    if collisionless:
        inset = axes[0].inset_axes([0.08, 0.18, 0.34, 0.34])
        inset.plot(
            [row["mode_count"] for row in collisionless],
            [row["growth"] for row in collisionless],
            "o-",
            color="#D97732",
            lw=1.3,
            ms=3.2,
        )
        inset.set(
            xlabel="modes",
            ylabel=r"$\gamma(\nu=0)$",
            title="collisionless endpoint",
        )
        inset.tick_params(labelsize=6.5)
        inset.xaxis.label.set_size(7)
        inset.yaxis.label.set_size(7)
        inset.title.set_size(7.5)
        inset.grid(alpha=0.2, lw=0.5)
        inset.spines[["top", "right"]].set_visible(False)

    for color, comparison in zip(colors[1:], summary["comparisons"], strict=True):
        upper = comparison["upper_resolution"]
        axes[1].plot(
            summary["curves"][0]["collision_frequency"],
            comparison["relative_growth_change"],
            "o-",
            color=color,
            lw=1.7,
            ms=3.5,
            label=rf"to $(P,J)=({upper[0]},{upper[1]})$",
        )
    tolerance = summary["thresholds"]["nested_growth_relative_tolerance"]
    axes[1].axhline(
        tolerance,
        color="#B33A3A",
        ls="--",
        lw=1.4,
        label=f"{100 * tolerance:g}% gate",
    )
    if not summary["gates"]["low_collisionality_growth_converged"]:
        axes[1].axvspan(
            0.0,
            summary["protocol"]["resolved_collision_minimum"],
            color="#D97732",
            alpha=0.12,
            label=r"low-$\nu$ unresolved",
        )
    axes[1].set_xscale("symlog", linthresh=1.0e-4)
    axes[1].set_xlim(0.0, 11.0)
    axes[1].set_yscale("log")
    axes[1].set(
        xlabel=r"collision frequency $\nu$",
        ylabel="nested relative growth change",
        title="(b) Velocity-space convergence",
    )
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    for axis in axes:
        axis.set_facecolor("white")
        axis.grid(alpha=0.22, lw=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    status = "CONVERGED" if summary["gate_passed"] else "OPEN"
    fig.suptitle(
        rf"Finite-wavelength Coulomb slab ITG at $k_\perp=0.5$ ({status})",
        fontsize=14,
        color="#20272E",
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_png,
        dpi=220,
        facecolor="white",
        metadata={"Software": "GKX"},
    )
    plt.close(fig)


def write_finite_wavelength_itg_artifacts(
    table_paths: list[Path], out_json: Path, out_png: Path
) -> dict[str, Any]:
    """Run exact tables through the slab RHS and write compact evidence."""

    collision_frequencies = np.concatenate(([0.0], np.logspace(-4.0, 1.0, 31)))
    curves = [
        finite_wavelength_itg_growth_curve(path, collision_frequencies)
        for path in table_paths
    ]
    collisionless = collisionless_slab_itg_hierarchy(
        ((4, 1), (6, 2), (8, 4), (10, 4), (10, 6), (12, 5), (15, 6), (18, 6))
    )
    summary = summarize_finite_wavelength_itg_curves(
        curves, collisionless_hierarchy=collisionless
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_finite_wavelength_itg_figure(summary, out_png)
    return summary


def build_collision_table(*, digits: int = 80) -> np.ndarray:
    """Generate the published C6/C9/103 matrices with multiprecision arithmetic."""

    import mpmath as mp

    with mp.workdps(digits):
        inverse_sqrt_pi = 1 / mp.sqrt(mp.pi)
        sqrt_two = mp.sqrt(2)
        sqrt_three = mp.sqrt(3)
        blocks = (
            (
                (-64 * sqrt_two / 45, 64 / 45, -32 * sqrt_two / 45),
                (
                    -361 * sqrt_two / 175,
                    208 / (175 * sqrt_three),
                    -1187 * sqrt_two / 525,
                ),
            ),
            (
                (-16 * sqrt_two / 15, 16 / 15, -8 * sqrt_two / 15),
                (-8 * sqrt_two / 5, 8 / (5 * sqrt_three), -28 * sqrt_two / 15),
            ),
        )
        matrices = np.zeros((3, 8, 8), dtype=np.float64)
        temperature_modes = (4, 1)
        heat_modes = (6, 3)
        for model, (thermal, heat) in zip((0, 2), blocks):
            for modes, coefficients in (
                (temperature_modes, thermal),
                (heat_modes, heat),
            ):
                row0, row1 = modes
                diagonal0, coupling, diagonal1 = coefficients
                matrices[model, row0, row0] = float(diagonal0 * inverse_sqrt_pi)
                matrices[model, row0, row1] = float(coupling * inverse_sqrt_pi)
                matrices[model, row1, row0] = float(coupling * inverse_sqrt_pi)
                matrices[model, row1, row1] = float(diagonal1 * inverse_sqrt_pi)
        matrices[1] = matrices[0]
        matrices[1, 6, 6] += float(mp.mpf(9) / 25 * mp.sqrt(2 / mp.pi))
        improved_heat_coupling = float(mp.mpf(6) / 25 * mp.sqrt(3 / mp.pi))
        matrices[1, 6, 3] += improved_heat_coupling
        matrices[1, 3, 6] += improved_heat_coupling
        matrices[1, 3, 3] += float(mp.mpf(6) / 25 * mp.sqrt(2 / mp.pi))
    return matrices


def write_collision_table(
    out: Path, metadata_out: Path, *, digits: int = 80
) -> dict[str, Any]:
    matrices = build_collision_table(digits=digits)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as stream:
        np.save(stream, matrices, allow_pickle=False)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    metadata = {
        "kind": "gkx_collision_moment_coefficients",
        "models": ["sugama", "improved_sugama", "coulomb"],
        "shape": list(matrices.shape),
        "dtype": str(matrices.dtype),
        "sha256": digest,
        "precision_decimal_digits": int(digits),
        "moment_order": "hermite_major_index=p*Nl+j",
        "Nl": 2,
        "Nm": 4,
        "laguerre_convention": "gkx_opposite_to_paper",
        "source": "Frei, Ernst & Ricci (2022), arXiv:2202.06293",
        "equations": {
            "sugama": "C6a-C6f",
            "improved_sugama": "C6a-C6f plus C103a-C103c",
            "coulomb": "C9a-C9f",
        },
        "claim_scope": "validated_drift_kinetic_like_species_low_order_vertical_slice",
    }
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def build_collision_table_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate collision coefficient tables."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_COLLISION_TABLE)
    parser.add_argument("--metadata-out", type=Path, default=DEFAULT_COLLISION_METADATA)
    parser.add_argument("--digits", type=int, default=80)
    return parser


def build_collision_verification_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the finite-b Coulomb operator verification artifacts."
    )
    parser.add_argument(
        "--out-json", type=Path, default=DEFAULT_COLLISION_VERIFICATION_JSON
    )
    parser.add_argument(
        "--out-png", type=Path, default=DEFAULT_COLLISION_VERIFICATION_PNG
    )
    parser.add_argument("--digits", type=int, default=80)
    return parser


def build_collision_response_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the drift-kinetic Coulomb response convergence artifacts."
    )
    parser.add_argument(
        "--out-json", type=Path, default=DEFAULT_COLLISION_RESPONSE_JSON
    )
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_COLLISION_RESPONSE_CSV)
    parser.add_argument("--out-png", type=Path, default=DEFAULT_COLLISION_RESPONSE_PNG)
    parser.add_argument("--digits", type=int, default=50)
    return parser


def build_collision_itg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate finite-wavelength slab-ITG convergence artifacts."
    )
    parser.add_argument(
        "--table",
        action="append",
        type=Path,
        required=True,
        help="Exact finite-wavelength pair-table archive; repeat by resolution.",
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_COLLISION_ITG_JSON)
    parser.add_argument("--out-png", type=Path, default=DEFAULT_COLLISION_ITG_PNG)
    return parser


def build_collision_endpoint_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one fixed-wavelength equal-species Coulomb archive."
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bessel-argument", type=float, required=True)
    parser.add_argument("--maximum-hermite-order", type=int, required=True)
    parser.add_argument("--maximum-laguerre-order", type=int, required=True)
    parser.add_argument("--maximum-angular-bessel-order", type=int, default=4)
    parser.add_argument("--maximum-bessel-laguerre-order", type=int, default=6)
    parser.add_argument("--digits", type=int, default=32)
    parser.add_argument("--worker-count", type=int, default=1)
    return parser


def build_collision_combine_angular_shards_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Combine complete finite-wavelength angular table shards."
    )
    parser.add_argument("--shard", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def build_collision_combine_wavelength_tables_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Combine complete angular tables on an ordered wavelength grid."
    )
    parser.add_argument("--table", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def build_collision_original_sugama_table_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an equal-species original-Sugama table from Coulomb tests."
    )
    parser.add_argument("--coulomb-table", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quadrature-order", type=int, default=80)
    return parser


def build_collision_project_table_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Project a finite-wavelength table onto a nested moment basis."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--maximum-hermite-order", type=int, required=True)
    parser.add_argument("--maximum-laguerre-order", type=int, required=True)
    return parser


def build_collision_improved_sugama_table_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an equal-species improved-Sugama table from Coulomb tests."
    )
    parser.add_argument("--coulomb-table", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--correction-order", type=int, default=5)
    parser.add_argument("--quadrature-order", type=int, default=80)
    parser.add_argument("--digits", type=int, default=80)
    return parser


def build_collision_shared_angular_table_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and combine angular shards from one shared speed cache."
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bessel-argument", action="append", type=float, required=True)
    parser.add_argument("--maximum-hermite-order", type=int, required=True)
    parser.add_argument("--maximum-laguerre-order", type=int, required=True)
    parser.add_argument("--maximum-angular-bessel-order", type=int, default=4)
    parser.add_argument("--maximum-bessel-laguerre-order", type=int, default=6)
    parser.add_argument("--digits", type=int, default=32)
    parser.add_argument("--worker-count", type=int, default=30)
    parser.add_argument("--wavelength-worker-count", type=int, default=2)
    return parser


def build_collision_diagonal_table_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an equal-species diagonal finite-wavelength table."
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bessel-argument", action="append", type=float, required=True)
    parser.add_argument("--maximum-hermite-order", type=int, required=True)
    parser.add_argument("--maximum-laguerre-order", type=int, required=True)
    parser.add_argument("--maximum-angular-bessel-order", type=int, default=4)
    parser.add_argument("--angular-order", action="append", type=int)
    parser.add_argument("--maximum-bessel-laguerre-order", type=int, default=6)
    parser.add_argument("--digits", type=int, default=32)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--wavelength-worker-count", type=int, default=1)
    return parser


def build_collision_contraction_gate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gate the fast finite-wavelength final contraction."
    )
    parser.add_argument("--exact-archive", type=Path, required=True)
    parser.add_argument("--fast-archive", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--maximum-relative-l2", type=float, default=2.0e-14)
    parser.add_argument("--maximum-absolute-error", type=float, default=2.0e-13)
    parser.add_argument("--minimum-speedup", type=float, default=1.25)
    return parser


def build_figures_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate linear validation figures.")
    parser.add_argument(
        "--case",
        choices=["all", "cyclone", "etg"],
        default="all",
        help="Limit figure generation to a specific case.",
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable progress bars."
    )
    return parser


def _load_gkx_scan_from_mismatch(
    csv_path: Path, *, x_col: str = "ky"
) -> LinearScanResult:
    from gkx.benchmarking.shared import LinearScanResult

    table = pd.read_csv(csv_path).sort_values(x_col)
    return LinearScanResult(
        ky=table[x_col].to_numpy(dtype=float),
        gamma=table["gamma_gkx"].to_numpy(dtype=float),
        omega=table["omega_gkx"].to_numpy(dtype=float),
    )


def _cyclone_refresh_reference(ref: LinearScanResult) -> LinearScanResult:
    from gkx.benchmarking.shared import LinearScanResult

    keep = np.asarray(ref.ky) <= 0.45 + 1.0e-12
    return LinearScanResult(
        ky=np.asarray(ref.ky)[keep],
        gamma=np.asarray(ref.gamma)[keep],
        omega=np.asarray(ref.omega)[keep],
    )


def _run_etg_figures(*, outdir: Path, progress: bool) -> None:
    from gkx.artifacts.plotting import scan_comparison_figure
    from gkx.benchmarking.shared import load_etg_reference
    from gkx.runtime import run_runtime_scan
    from gkx.workflows.runtime.toml import load_runtime_from_toml

    reference = load_etg_reference()
    mismatch_csv = outdir / "etg_mismatch_table.csv"
    if mismatch_csv.exists():
        scan = _load_gkx_scan_from_mismatch(mismatch_csv)
    else:
        config, _ = load_runtime_from_toml(
            REPO_ROOT / "examples/linear/axisymmetric/etg.toml"
        )
        scan = run_runtime_scan(
            config,
            np.asarray(reference.ky),
            Nl=24,
            Nm=8,
            solver="time",
            batch_ky=True,
            method=config.time.method,
            dt=config.time.dt,
            steps=int(round(config.time.t_max / config.time.dt)),
            sample_stride=config.time.sample_stride,
            auto_window=False,
            tmin=1.0,
            tmax=config.time.t_max,
            fit_signal="phi",
            mode_method="z_index",
            show_progress=progress,
        )
    fig, _axes = scan_comparison_figure(
        scan.ky,
        scan.gamma,
        scan.omega,
        r"$k_y \rho_i$",
        "ETG Benchmark Scan",
        x_ref=reference.ky,
        gamma_ref=reference.gamma,
        omega_ref=reference.omega,
        label="GKX",
        ref_label="Reference",
        log_x=True,
    )
    fig.savefig(outdir / "etg_comparison.png", dpi=200)
    fig.savefig(outdir / "etg_comparison.pdf")


def _run_kbm_figures(*, outdir: Path) -> None:
    from gkx.artifacts.plotting import scan_comparison_figure
    from gkx.benchmarking.shared import load_kbm_reference

    reference = load_kbm_reference()
    mismatch_csv = outdir / "kbm_mismatch_table.csv"
    if not mismatch_csv.exists():
        raise FileNotFoundError(
            f"missing {mismatch_csv}; generate the KBM mismatch table first"
        )
    scan = _load_gkx_scan_from_mismatch(mismatch_csv)
    fig, _axes = scan_comparison_figure(
        scan.ky,
        scan.gamma,
        scan.omega,
        r"$\beta$",
        "KBM Benchmark Scan",
        x_ref=reference.ky,
        gamma_ref=reference.gamma,
        omega_ref=reference.omega,
        label="GKX",
        ref_label="Reference",
        log_x=False,
    )
    fig.savefig(outdir / "kbm_comparison.png", dpi=200)
    fig.savefig(outdir / "kbm_comparison.pdf")


def main_figures(argv: list[str] | None = None) -> int:
    from gkx.artifacts.plotting import (
        cyclone_comparison_figure,
        cyclone_reference_figure,
    )
    from gkx.benchmarking.shared import load_cyclone_reference

    args = build_figures_parser().parse_args(argv)
    outdir = REPO_ROOT / "docs" / "_static"
    outdir.mkdir(parents=True, exist_ok=True)
    progress = not args.no_progress
    if args.case == "etg":
        _run_etg_figures(outdir=outdir, progress=progress)
        return 0

    reference = _cyclone_refresh_reference(load_cyclone_reference())
    fig, _axes = cyclone_reference_figure(reference)
    fig.savefig(outdir / "cyclone_reference.png", dpi=200)
    fig.savefig(outdir / "cyclone_reference.pdf")
    scan = _load_gkx_scan_from_mismatch(outdir / "cyclone_mismatch_table.csv")
    fig, _axes = cyclone_comparison_figure(reference, scan)
    fig.savefig(outdir / "cyclone_comparison.png", dpi=200)
    fig.savefig(outdir / "cyclone_comparison.pdf")
    if args.case == "cyclone":
        return 0

    _run_etg_figures(outdir=outdir, progress=progress)
    _run_kbm_figures(outdir=outdir)
    return 0


def _json_clean(value: Any) -> Any:
    """Return a strict-JSON-compatible copy with nonfinite numbers set to null."""

    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_json_clean(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def load_convergence_series(
    csv_path: Path,
    *,
    step_column: str | None,
    resolution_column: str | None,
    error_column: str,
    absolute_error: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Load effective step sizes and errors from a convergence table."""

    if (step_column is None) == (resolution_column is None):
        raise ValueError("Specify exactly one of step_column or resolution_column.")
    table = pd.read_csv(csv_path)
    required = {error_column, step_column or resolution_column}
    missing = sorted(col for col in required if col not in table.columns)
    if missing:
        raise ValueError(
            f"{csv_path} is missing required columns: {', '.join(missing)}"
        )

    if step_column is not None:
        h = np.asarray(table[step_column], dtype=float)
        step_source = step_column
    else:
        resolution = np.asarray(table[resolution_column], dtype=float)
        if np.any(resolution <= 0.0):
            raise ValueError("resolution values must be positive")
        h = 1.0 / resolution
        step_source = f"1/{resolution_column}"

    err = np.asarray(table[error_column], dtype=float)
    if absolute_error:
        err = np.abs(err)

    order = np.argsort(h)[::-1]
    h = h[order]
    err = err[order]
    selected = table.iloc[order].copy()
    selected["effective_step"] = h
    selected["error_for_gate"] = err
    selected["step_source"] = step_source
    rows = _json_clean(selected.to_dict(orient="records"))
    return h, err, rows


def build_summary(
    csv_path: Path,
    *,
    step_column: str | None,
    resolution_column: str | None,
    error_column: str,
    case: str,
    source: str,
    min_order: float,
    min_pairwise_order: float | None,
    max_final_error: float | None,
    absolute_error: bool = True,
) -> dict[str, object]:
    """Build the JSON payload for an observed-order convergence gate."""

    from gkx.diagnostics.analysis import estimate_observed_order
    from gkx.diagnostics.validation_gates import (
        gate_report_to_dict,
        observed_order_gate_report,
    )

    h, err, rows = load_convergence_series(
        csv_path,
        step_column=step_column,
        resolution_column=resolution_column,
        error_column=error_column,
        absolute_error=absolute_error,
    )
    metrics = estimate_observed_order(h, err)
    report = observed_order_gate_report(
        metrics,
        case=case,
        source=source,
        min_asymptotic_order=min_order,
        min_pairwise_order=min_pairwise_order,
        max_final_error=max_final_error,
    )
    payload = {
        "case": case,
        "source": source,
        "csv": str(csv_path),
        "error_column": error_column,
        "absolute_error": bool(absolute_error),
        "step_sizes": metrics.step_sizes.tolist(),
        "errors": metrics.errors.tolist(),
        "pairwise_orders": metrics.orders.tolist(),
        "asymptotic_order": metrics.asymptotic_order,
        "min_pairwise_order": float(np.min(metrics.orders)),
        "final_error": float(metrics.errors[-1]),
        "gate_report": gate_report_to_dict(report),
        "gate_passed": bool(report.passed),
        "rows": rows,
    }
    return _json_clean(payload)


def write_observed_order_plot(
    summary: dict[str, object],
    out_png: Path,
    *,
    title: str,
    min_order: float,
) -> None:
    """Write a log-log convergence panel for an observed-order gate."""

    h = np.asarray(summary["step_sizes"], dtype=float)
    err = np.asarray(summary["errors"], dtype=float)
    asymptotic_order = float(summary["asymptotic_order"])
    min_pairwise_order = float(summary["min_pairwise_order"])
    final_error = float(summary["final_error"])
    gate_status = "passed" if bool(summary["gate_passed"]) else "open"

    fig, ax = plt.subplots(figsize=(5.2, 3.7), constrained_layout=True)
    ax.loglog(h, err, "o-", color="#1b6ca8", lw=2.0, ms=6, label="measured error")
    ref = err[-1] * (h / h[-1]) ** float(min_order)
    ax.loglog(h, ref, "--", color="#8f2d2d", lw=1.8, label=f"order {min_order:g} guide")
    ax.invert_xaxis()
    ax.set_xlabel("effective step size h (coarse to fine)")
    ax.set_ylabel("absolute error")
    ax.set_title(title, fontsize=13)
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(frameon=False, loc="best")
    ax.text(
        0.03,
        0.04,
        (
            f"final-pair order = {asymptotic_order:.2f}\n"
            f"min pairwise order = {min_pairwise_order:.2f}\n"
            f"final error = {final_error:.3g}\n"
            f"gate: {gate_status}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.85", "alpha": 0.92},
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def _branch_gate_report_from_selected_rows(
    rows: list[dict[str, object]],
    *,
    max_rel_gamma_jump: float,
    max_rel_omega_jump: float,
    min_successive_overlap: float | None,
) -> dict[str, object] | None:
    from gkx.diagnostics.analysis import branch_continuity_metrics
    from gkx.diagnostics.validation_gates import (
        branch_continuity_gate_report,
        gate_report_to_dict,
    )

    if len(rows) < 2:
        return None
    table = pd.DataFrame(rows).sort_values("ky")
    overlap = None
    if "eig_overlap_prev" in table.columns:
        overlap_values = np.asarray(table["eig_overlap_prev"], dtype=float)[1:]
        if overlap_values.size == table.shape[0] - 1 and np.all(
            np.isfinite(overlap_values)
        ):
            overlap = overlap_values
    metrics = branch_continuity_metrics(
        np.asarray(table["ky"], dtype=float),
        np.asarray(table["gamma"], dtype=float),
        np.asarray(table["omega"], dtype=float),
        successive_overlap=overlap,
    )
    report = branch_continuity_gate_report(
        metrics,
        case="kbm_linear_branch_continuity",
        source="selected KBM comparison rows",
        max_rel_gamma_jump=max_rel_gamma_jump,
        max_rel_omega_jump=max_rel_omega_jump,
        min_successive_overlap=(
            min_successive_overlap if overlap is not None else None
        ),
    )
    return gate_report_to_dict(report)


def _coerce_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def selected_candidate_rows(path: Path) -> list[dict[str, object]]:
    """Load selected branch rows from a KBM candidate table."""

    table = pd.read_csv(path)
    required = {"ky", "gamma", "omega", "selected"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    selected = table["selected"].map(_coerce_bool)
    rows = table.loc[selected].sort_values("ky").to_dict(orient="records")
    return [_json_clean(row) for row in rows]


def build_kbm_branch_summary(
    candidate_csv: Path,
    *,
    max_rel_gamma_jump: float,
    max_rel_omega_jump: float,
    min_successive_overlap: float | None,
) -> dict[str, object]:
    """Build the JSON payload for the selected KBM branch-continuity gate."""

    rows = selected_candidate_rows(candidate_csv)
    try:
        candidate_label = str(candidate_csv.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        candidate_label = str(candidate_csv)
    report = _branch_gate_report_from_selected_rows(
        rows,
        max_rel_gamma_jump=max_rel_gamma_jump,
        max_rel_omega_jump=max_rel_omega_jump,
        min_successive_overlap=min_successive_overlap,
    )
    payload: dict[str, object] = {
        "case": "kbm_linear_branch_continuity",
        "candidate_csv": candidate_label,
        "selected_count": len(rows),
        "thresholds": {
            "max_rel_gamma_jump": float(max_rel_gamma_jump),
            "max_rel_omega_jump": float(max_rel_omega_jump),
            "min_successive_overlap": min_successive_overlap,
        },
        "rows": rows,
        "gate_report": report,
        "gate_passed": None if report is None else bool(report["passed"]),
        "notes": (
            "Selected rows are taken from the KBM comparison candidate table. "
            "The gate is intentionally a branch-identity check: adjacent gamma "
            "and omega jumps should stay smooth, and successive eigenfunction "
            "overlaps should remain high when those overlaps are available."
        ),
    }
    return _json_clean(payload)


def build_observed_order_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write an observed-order gate report from a convergence CSV."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_OBSERVED_CSV)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--step-column", default=None)
    group.add_argument("--resolution-column", default=None)
    parser.add_argument("--error-column", default="rel_gamma")
    parser.add_argument("--case", default="cyclone_resolution_observed_order")
    parser.add_argument("--source", default="tracked Cyclone resolution subset")
    parser.add_argument("--min-order", type=float, default=1.0)
    parser.add_argument(
        "--min-pairwise-order",
        type=float,
        default=0.0,
        help="Optional floor for every pairwise observed order; set negative to disable.",
    )
    parser.add_argument("--max-final-error", type=float, default=0.05)
    parser.add_argument(
        "--signed-error",
        action="store_true",
        help="Use signed errors instead of absolute values.",
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OBSERVED_JSON)
    parser.add_argument("--out-png", type=Path, default=DEFAULT_OBSERVED_PNG)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--title", default="Cyclone Resolution Convergence")
    return parser


def build_kbm_branch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a KBM branch-continuity gate summary from selected candidate rows."
    )
    parser.add_argument("--candidates", type=Path, default=DEFAULT_KBM_CANDIDATES)
    parser.add_argument("--out", type=Path, default=DEFAULT_KBM_BRANCH_OUT)
    parser.add_argument("--max-rel-gamma-jump", type=float, default=0.5)
    parser.add_argument("--max-rel-omega-jump", type=float, default=0.5)
    parser.add_argument("--min-successive-overlap", type=float, default=0.95)
    return parser


def main_observed_order(argv: list[str] | None = None) -> int:
    args = build_observed_order_parser().parse_args(argv)
    resolution_column = (
        args.resolution_column if args.resolution_column is not None else "Nm"
    )
    min_pairwise_order = (
        None
        if args.min_pairwise_order is None or float(args.min_pairwise_order) < 0.0
        else float(args.min_pairwise_order)
    )
    summary = build_summary(
        args.csv,
        step_column=args.step_column,
        resolution_column=resolution_column if args.step_column is None else None,
        error_column=args.error_column,
        case=args.case,
        source=args.source,
        min_order=float(args.min_order),
        min_pairwise_order=min_pairwise_order,
        max_final_error=args.max_final_error,
        absolute_error=not bool(args.signed_error),
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    if not args.no_plot:
        write_observed_order_plot(
            summary, args.out_png, title=args.title, min_order=float(args.min_order)
        )
    print(f"Wrote {args.out_json}")
    if not args.no_plot:
        print(f"Wrote {args.out_png}")
    return 0


def main_kbm_branch(argv: list[str] | None = None) -> int:
    args = build_kbm_branch_parser().parse_args(argv)
    summary = build_kbm_branch_summary(
        args.candidates,
        max_rel_gamma_jump=float(args.max_rel_gamma_jump),
        max_rel_omega_jump=float(args.max_rel_omega_jump),
        min_successive_overlap=float(args.min_successive_overlap),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(f"Wrote {args.out}")
    return 0


def main_collision_table(argv: list[str] | None = None) -> int:
    args = build_collision_table_parser().parse_args(argv)
    metadata = write_collision_table(
        args.out, args.metadata_out, digits=int(args.digits)
    )
    print(f"Wrote {args.out} ({metadata['sha256']})")
    print(f"Wrote {args.metadata_out}")
    return 0


def main_collision_verification(argv: list[str] | None = None) -> int:
    args = build_collision_verification_parser().parse_args(argv)
    summary = write_coulomb_operator_verification_artifacts(
        args.out_json,
        args.out_png,
        digits=int(args.digits),
    )
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_png}")
    print(
        f"Collision verification gate: {'PASS' if summary['gate_passed'] else 'FAIL'}"
    )
    return 0 if summary["gate_passed"] else 1


def main_collision_response(argv: list[str] | None = None) -> int:
    args = build_collision_response_parser().parse_args(argv)
    summary = write_drift_kinetic_response_convergence_artifacts(
        args.out_json,
        args.out_csv,
        args.out_png,
        digits=int(args.digits),
    )
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_png}")
    print(f"Collision response gate: {'PASS' if summary['gate_passed'] else 'FAIL'}")
    return 0 if summary["gate_passed"] else 1


def main_collision_itg(argv: list[str] | None = None) -> int:
    args = build_collision_itg_parser().parse_args(argv)
    summary = write_finite_wavelength_itg_artifacts(
        args.table, args.out_json, args.out_png
    )
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_png}")
    print(
        "Finite-wavelength ITG literature gate: "
        f"{'PASS' if summary['gate_passed'] else 'OPEN'}"
    )
    return 0


def main_collision_endpoint(argv: list[str] | None = None) -> int:
    args = build_collision_endpoint_parser().parse_args(argv)
    metadata = write_finite_wavelength_coulomb_endpoint(
        args.out,
        bessel_argument=float(args.bessel_argument),
        maximum_hermite_order=int(args.maximum_hermite_order),
        maximum_laguerre_order=int(args.maximum_laguerre_order),
        maximum_angular_bessel_order=int(args.maximum_angular_bessel_order),
        maximum_bessel_laguerre_order=int(args.maximum_bessel_laguerre_order),
        digits=int(args.digits),
        worker_count=int(args.worker_count),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


def main_collision_combine_angular_shards(argv: list[str] | None = None) -> int:
    args = build_collision_combine_angular_shards_parser().parse_args(argv)
    metadata = combine_equal_species_finite_wavelength_angular_shards(
        tuple(args.shard), args.out
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


def main_collision_combine_wavelength_tables(argv: list[str] | None = None) -> int:
    args = build_collision_combine_wavelength_tables_parser().parse_args(argv)
    metadata = combine_equal_species_finite_wavelength_tables(
        tuple(args.table), args.out
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


def main_collision_original_sugama_table(argv: list[str] | None = None) -> int:
    args = build_collision_original_sugama_table_parser().parse_args(argv)
    metadata = write_equal_species_finite_wavelength_original_sugama_table(
        args.coulomb_table,
        args.out,
        quadrature_order=int(args.quadrature_order),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


def main_collision_project_table(argv: list[str] | None = None) -> int:
    args = build_collision_project_table_parser().parse_args(argv)
    metadata = project_equal_species_finite_wavelength_table(
        args.source,
        args.out,
        maximum_hermite_order=int(args.maximum_hermite_order),
        maximum_laguerre_order=int(args.maximum_laguerre_order),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


def main_collision_improved_sugama_table(argv: list[str] | None = None) -> int:
    args = build_collision_improved_sugama_table_parser().parse_args(argv)
    metadata = write_equal_species_finite_wavelength_improved_sugama_table(
        args.coulomb_table,
        args.out,
        correction_order=int(args.correction_order),
        quadrature_order=int(args.quadrature_order),
        digits=int(args.digits),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


def main_collision_shared_angular_table(argv: list[str] | None = None) -> int:
    args = build_collision_shared_angular_table_parser().parse_args(argv)
    metadata = write_shared_precompute_angular_coulomb_table(
        args.out,
        bessel_arguments=tuple(args.bessel_argument),
        maximum_hermite_order=int(args.maximum_hermite_order),
        maximum_laguerre_order=int(args.maximum_laguerre_order),
        maximum_angular_bessel_order=int(args.maximum_angular_bessel_order),
        maximum_bessel_laguerre_order=int(args.maximum_bessel_laguerre_order),
        digits=int(args.digits),
        worker_count=int(args.worker_count),
        wavelength_worker_count=int(args.wavelength_worker_count),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


def main_collision_diagonal_table(argv: list[str] | None = None) -> int:
    args = build_collision_diagonal_table_parser().parse_args(argv)
    metadata = write_equal_species_finite_wavelength_coulomb_table(
        args.out,
        bessel_arguments=tuple(args.bessel_argument),
        maximum_hermite_order=int(args.maximum_hermite_order),
        maximum_laguerre_order=int(args.maximum_laguerre_order),
        maximum_angular_bessel_order=int(args.maximum_angular_bessel_order),
        maximum_bessel_laguerre_order=int(args.maximum_bessel_laguerre_order),
        included_angular_orders=(
            None if args.angular_order is None else tuple(sorted(args.angular_order))
        ),
        digits=int(args.digits),
        worker_count=int(args.worker_count),
        wavelength_worker_count=int(args.wavelength_worker_count),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


def main_collision_contraction_gate(argv: list[str] | None = None) -> int:
    args = build_collision_contraction_gate_parser().parse_args(argv)
    report = write_collision_table_contraction_gate(
        args.exact_archive,
        args.fast_archive,
        args.out_json,
        maximum_relative_l2=float(args.maximum_relative_l2),
        maximum_absolute_error=float(args.maximum_absolute_error),
        minimum_speedup=float(args.minimum_speedup),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["gate_passed"] else 1


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "command",
            choices=(
                "figures",
                "observed-order",
                "kbm-branch",
                "collision-table",
                "collision-verification",
                "collision-response",
                "collision-itg",
                "collision-endpoint",
                "collision-diagonal-table",
                "collision-combine-angular-shards",
                "collision-combine-wavelength-tables",
                "collision-original-sugama-table",
                "collision-improved-sugama-table",
                "collision-project-table",
                "collision-shared-angular-table",
                "collision-contraction-gate",
            ),
        )
        parser.print_help()
        return 2
    command, rest = tokens[0], tokens[1:]
    if command == "figures":
        return main_figures(rest)
    if command == "observed-order":
        return main_observed_order(rest)
    if command == "kbm-branch":
        return main_kbm_branch(rest)
    if command == "collision-table":
        return main_collision_table(rest)
    if command == "collision-verification":
        return main_collision_verification(rest)
    if command == "collision-response":
        return main_collision_response(rest)
    if command == "collision-itg":
        return main_collision_itg(rest)
    if command == "collision-endpoint":
        return main_collision_endpoint(rest)
    if command == "collision-diagonal-table":
        return main_collision_diagonal_table(rest)
    if command == "collision-combine-angular-shards":
        return main_collision_combine_angular_shards(rest)
    if command == "collision-combine-wavelength-tables":
        return main_collision_combine_wavelength_tables(rest)
    if command == "collision-original-sugama-table":
        return main_collision_original_sugama_table(rest)
    if command == "collision-improved-sugama-table":
        return main_collision_improved_sugama_table(rest)
    if command == "collision-project-table":
        return main_collision_project_table(rest)
    if command == "collision-shared-angular-table":
        return main_collision_shared_angular_table(rest)
    if command == "collision-contraction-gate":
        return main_collision_contraction_gate(rest)
    raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
