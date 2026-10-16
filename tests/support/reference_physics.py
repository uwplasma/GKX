"""Reference physics helpers shared by validation tests.

Extracted from evidence generators that were removed from ``scripts/``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any
import numpy as np
import jax
import jax.numpy as jnp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from scipy.integrate import solve_ivp
from scipy.special import wofz
from scipy.optimize import curve_fit, fsolve
from gkx.config import GeometryConfig, GridConfig
from gkx.core_grid import build_spectral_grid
from gkx.geometry import SlabGeometry
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import LinearTerms, linear_params_for_geometry
from gkx.operators.linear.rhs import linear_rhs_cached
from gkx.terms.linear_terms import hermite_closure_coefficient
from gkx.workflows.runtime.toml import load_runtime_from_toml
from gkx.operators.linear.params import (
    LinearParams,
)
from gkx.operators.linear.params import build_linear_params
from typing import Sequence
from gkx.config import InitializationConfig
from gkx.core_grid import SpectralGrid
from gkx.geometry import FluxTubeGeometryLike
from gkx.operators.linear.params import Species


# --- from scripts/artifacts/build_linear_validation_artifacts.py (removed; see plan/log.md) ---
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


# --- from scripts/campaigns/vmec_state_sensitivity.py (removed; see plan/log.md) ---
def _array_parity_metrics(
    candidate: Any, reference: Any, *, floor: float = 1.0e-12
) -> dict[str, object]:
    cand = np.asarray(candidate, dtype=float)
    ref = np.asarray(reference, dtype=float)
    metrics: dict[str, object] = {
        "candidate_shape": [int(v) for v in cand.shape],
        "reference_shape": [int(v) for v in ref.shape],
        "shape_match": bool(cand.shape == ref.shape),
    }
    if cand.shape != ref.shape:
        return metrics
    diff = cand - ref
    ref_scale = max(float(np.nanmax(np.abs(ref))) if ref.size else 0.0, float(floor))
    local_scale = np.maximum(np.abs(ref), float(floor))
    metrics.update(
        {
            "max_abs": float(np.nanmax(np.abs(diff))) if diff.size else 0.0,
            "rms_abs": float(np.sqrt(np.nanmean(diff * diff))) if diff.size else 0.0,
            "max_rel_pointwise": float(np.nanmax(np.abs(diff) / local_scale))
            if diff.size
            else 0.0,
            "rms_rel_pointwise": float(np.sqrt(np.nanmean((diff / local_scale) ** 2)))
            if diff.size
            else 0.0,
            "reference_scale": ref_scale,
            "normalized_max_abs": float(np.nanmax(np.abs(diff)) / ref_scale)
            if diff.size
            else 0.0,
            "candidate_min": float(np.nanmin(cand)) if cand.size else 0.0,
            "candidate_max": float(np.nanmax(cand)) if cand.size else 0.0,
            "reference_min": float(np.nanmin(ref)) if ref.size else 0.0,
            "reference_max": float(np.nanmax(ref)) if ref.size else 0.0,
        }
    )
    return metrics


def _scalar_parity_metrics(
    candidate: Any, reference: Any, *, floor: float = 1.0e-12
) -> dict[str, float]:
    cand = float(np.asarray(candidate))
    ref = float(np.asarray(reference))
    diff = cand - ref
    scale = max(abs(ref), float(floor))
    return {
        "candidate": cand,
        "reference": ref,
        "abs": abs(diff),
        "rel": abs(diff) / scale,
    }


# --- from scripts/artifacts/build_analytic_benchmarks.py (removed; see plan/log.md) ---
ROOT = Path(__file__).resolve().parents[2]

OUT = ROOT / "docs/_static/analytic_benchmarks.json"

ZONAL_DECK = ROOT / "benchmarks/cases/miller_zonal_response.toml"

EM_DECK = ROOT / "examples/06_electromagnetic/case_full.toml"

R_OVER_A = 2.77778

M_E = 1.0 / 1836.15


def rosenbluth_hinton(q: float, eps: float) -> float:
    """Rosenbluth & Hinton, PRL 80, 724 (1998): 1/(1 + 1.6 q^2/sqrt(eps))."""
    return 1.0 / (1.0 + 1.6 * q**2 / np.sqrt(eps))


def xiao_catto(
    q: float, eps: float, kappa: float = 1.0, delta: float = 0.0, shift: float = 0.0
) -> float:
    """Xiao & Catto, PoP 13, 082307 (2006), Eqs. (38)-(39)."""
    s = (
        3.27
        + np.sqrt(eps)
        + 0.722 * eps
        - 1.443 * delta
        - 2.945 * shift / eps
        + (0.692 * kappa**2 - 0.722) / q**2 * eps
    ) / (1.0 + kappa**2)
    return 1.0 / (1.0 + s * q**2 / np.sqrt(eps))


def _zfun(z: complex) -> complex:
    return 1j * np.sqrt(np.pi) * wofz(z)


def sw_gam_formula(
    q: float, tau_e: float = 1.0, krai: float = 0.0
) -> tuple[float, float]:
    """Sugama & Watanabe, JPP 72, 825 (2006), Eqs. (2.9)-(2.10).

    Returns (omega_G, gamma) in v_ti/R0 (GKX units); tau_e = T_e/T_i and
    krai = k_r rho_i (GKX rho_i = sqrt(T_i/m_i)/Omega_i).
    """
    w = np.sqrt(7 + 4 * tau_e) / 2
    w *= np.sqrt(
        1 + 2 * (23 + 16 * tau_e + 4 * tau_e**2) / (q**2 * (7 + 4 * tau_e) ** 2)
    )
    wh, kq2 = q * w, 2 * (krai * q) ** 2
    corr = 1 + 2 * (23 / 4 + 4 * tau_e + tau_e**2) / (q**2 * (7 / 2 + 2 * tau_e) ** 2)
    br = np.exp(-(wh**2)) * (wh**4 + (1 + 2 * tau_e) * wh**2) + 0.25 * kq2 * np.exp(
        -(wh**2) / 4
    ) * (wh**6 / 64 + (1 + 3 * tau_e / 8) * (wh**4 / 8 + 3 * wh**2 / 4))
    return float(np.sqrt(2) * w), float(-np.sqrt(2 * np.pi) / 2 * q * br / corr)


def _sw_inverse_k(wh: complex, q: float, tau_e: float, krai: float) -> complex:
    """SW Eq. (2.7); wh = R0 q omega / sqrt(2 T_i/m_i). Z_r is read as Re Z."""
    z = _zfun(wh)
    a = 2 * wh + (2 * wh**2 + 1) * z
    d = 1.0 / tau_e + 1 + wh * z
    main = 2 * wh**3 + 3 * wh + (2 * wh**4 + 2 * wh**2 + 1) * z - wh / 2 * a**2 / d
    wr = wh.real
    zr = _zfun(wr)
    ar, dr = 2 * wr + (2 * wr**2 + 1) * zr, 1.0 / tau_e + 1 + wr * zr.real
    orbit = wr**6 / 64 + (wr**4 / 8 + 3 * wr**2 / 4 + 3 + 6 / wr**2) * (
        1 - 3 * wr / 16 * ar / dr
    )
    main += 1j * np.sqrt(np.pi) / 2 * 2 * (krai * q) ** 2 * np.exp(-(wr**2) / 4) * orbit
    return -1j * wh - 1j * q**2 / 2 * main


def sw_gam_root(q: float, tau_e: float = 1.0, krai: float = 0.0) -> tuple[float, float]:
    """Root of SW Eq. (2.7), (omega_G, gamma) in v_ti/R0 (GKX units)."""
    w, g = (x / np.sqrt(2) for x in sw_gam_formula(q, tau_e, krai))
    x0, x1 = q * complex(w, g), q * complex(1.01 * w, g)
    f0, f1 = _sw_inverse_k(x0, q, tau_e, krai), _sw_inverse_k(x1, q, tau_e, krai)
    for _ in range(100):
        x0, x1 = x1, x1 - f1 * (x1 - x0) / (f1 - f0)
        f0, f1 = f1, _sw_inverse_k(x1, q, tau_e, krai)
        if abs(x1 - x0) < 1e-12:
            break
    return float(np.sqrt(2) * x1.real / q), float(np.sqrt(2) * x1.imag / q)


def _ballooning_crosses(s: float, a: float, tmax: float) -> bool:
    """Even solution of CHT Eq. (10) has a zero on (0, tmax] (Newcomb: unstable)."""

    def rhs(t: float, y: np.ndarray) -> list[float]:
        h = s * t - a * np.sin(t)
        return [y[1] / (1 + h * h), -a * (np.cos(t) + h * np.sin(t)) * y[0]]

    def event(t: float, y: np.ndarray) -> float:
        return y[0]

    event.terminal = True  # type: ignore[attr-defined]
    sol = solve_ivp(
        rhs, [0.0, tmax], [1.0, 0.0], events=event, rtol=1e-10, atol=1e-13, max_step=0.5
    )
    return sol.t_events[0].size > 0


def cht_alpha_crit(
    s: float,
    tmax: float = 400.0,
    tol: float = 1e-4,
    bracket: tuple[float, float] | None = None,
) -> float:
    """Ideal s-alpha first-stability boundary, Connor-Hastie-Taylor PRL 40, 396.

    The truncated domain overestimates by O(1/tmax): 400 is within 2e-3 of the
    converged values in REPORT.md section 2.6 for 0.4 <= s <= 1.5. `bracket`
    (stable, unstable) skips the coarse scan.
    """
    if bracket is None:
        grid = np.linspace(0.05, 2.0, 40)
        i = next(k for k, a in enumerate(grid) if _ballooning_crosses(s, a, tmax))
        bracket = (grid[i - 1], grid[i])
    lo, hi = bracket
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if _ballooning_crosses(s, mid, tmax) else (mid, hi)
    return float(0.5 * (lo + hi))


def pkj_beta_mhd(s_hat: float, q: float, rln: float, rlti: float, rlte: float) -> float:
    """Pueschel-Kammerer-Jenko PoP 15, 102310 (2008): 0.6 s/(q^2 sum R/L)."""
    return 0.6 * s_hat / (q**2 * (2 * rln + rlti + rlte))


def fit_zonal_response(
    t: np.ndarray, phi: np.ndarray, tmin: float = 2.0
) -> tuple[float, float, float]:
    """Fit phi_inf + A cos(omega t + p) exp(gamma t) over t >= tmin.

    Returns (residual, omega_G, gamma) in the units of ``t``. Fitting the GAM
    and the residual together removes the bias of a tail mean over a few
    weakly damped GAM periods.
    """
    from scipy.optimize import curve_fit

    t, phi = np.asarray(t, float), np.asarray(phi, float)
    keep = t >= tmin
    spectrum = np.abs(np.fft.rfft(phi[keep] - phi[keep].mean()))
    omega0 = (
        2 * np.pi * np.fft.rfftfreq(keep.sum(), t[1] - t[0])[int(np.argmax(spectrum))]
    )

    def model(
        x: np.ndarray, res: float, amp: float, om: float, gam: float, ph: float
    ) -> np.ndarray:
        return res + amp * np.cos(om * x + ph) * np.exp(gam * (x - tmin))

    p0 = [float(phi[keep][-len(phi[keep]) // 3 :].mean()), 0.3, omega0, -0.05, 0.0]
    popt, _ = curve_fit(model, t[keep], phi[keep], p0=p0, maxfev=20000)
    return float(popt[0]), float(abs(popt[2])), float(popt[3])


def zonal_run(
    q: float,
    eps: float,
    kx: float,
    *,
    model: str = "s-alpha",
    kappa: float = 1.0,
    Nm: int = 128,
    Nz: int = 32,
    t_max_R0: float = 22.0,
    dt: float = 2.5e-3,
    hyper: bool = False,
) -> dict:
    """ky = 0, Boltzmann electrons with the zonal correction, rhoc = 0.5 a.

    Returns the z-averaged zonal potential over its t = 0 value, t in R0/v_ti.
    `hyper` switches on the GX |k_z| Hermite hypercollisions, which absorb the
    free-streaming recurrence of a low-Nm run.
    """
    from gkx.runtime import run_runtime_linear

    cfg, _ = load_runtime_from_toml(ZONAL_DECK)
    r0 = 0.5 / eps
    geo = replace(
        cfg.geometry, model=model, q=q, s_hat=0.8, epsilon=eps, R0=r0, rhoc=0.5
    )
    if model == "miller":
        geo = replace(
            geo, R_geo=r0, shift=0.0, akappa=kappa, akappri=0.0, tri=0.0, tripri=0.0
        )
    cfg = replace(cfg, geometry=geo, grid=replace(cfg.grid, Nz=Nz, Lx=2 * np.pi / kx))
    if hyper:
        cfg = replace(
            cfg,
            physics=replace(cfg.physics, hypercollisions=True),
            terms=replace(cfg.terms, hypercollisions=1.0),
            collisions=replace(cfg.collisions, hypercollisions_kz=1.0),
        )
    stride = max(1, int(round(0.05 / dt)))
    res = run_runtime_linear(
        cfg,
        ky_target=0.0,
        kx_target=kx,
        Nl=4,
        Nm=Nm,
        solver="time",
        dt=dt,
        steps=stride * int(np.ceil(t_max_R0 * r0 / dt / stride)),
        sample_stride=stride,
        fit_signal="phi",
        require_positive=False,
        min_points=4,
    )
    field = np.asarray(res.field_history)[:, 0]
    k = int(np.argmax(np.abs(field[0]).max(axis=1)))
    trace = np.real(field[:, k].mean(axis=1))
    return {
        "t": (np.asarray(res.t) / r0).round(5).tolist(),
        "phi": (trace / trace[0]).round(6).tolist(),
    }


def em_run(
    ky: float,
    beta: float,
    tprim: float,
    *,
    fprim: float = 0.8,
    geom: str = "s-alpha",
    consistent: bool = False,
    bpar: bool = False,
    q: float = 1.4,
    s_hat: float = 0.786,
    eps: float = 0.18,
    Nm: int = 16,
    Nl: int = 4,
    nperiod: int = 3,
    t_max: float = 60.0,
    dt: float = 1e-3,
) -> dict:
    """Linear EM CBC-like point, kinetic hydrogen electrons, lengths in a (R0 = 2.78 a).

    `consistent` sets the s-alpha alpha (or Miller betaprim) from beta and the
    gradients; otherwise alpha = 0 as in Pueschel-Kammerer-Jenko.
    """
    from gkx.runtime import run_runtime_linear

    cfg, _ = load_runtime_from_toml(EM_DECK)
    species = tuple(
        replace(s, tprim=tprim, fprim=fprim, mass=M_E if s.charge < 0 else 1.0)
        for s in cfg.species
    )
    drive = 2 * (tprim + fprim)
    geo = replace(cfg.geometry, q=q, s_hat=s_hat, epsilon=eps, R0=R_OVER_A)
    if geom == "miller":
        geo = replace(
            geo,
            model="miller",
            R_geo=R_OVER_A,
            shift=0.0,
            akappa=1.0,
            akappri=0.0,
            tri=0.0,
            tripri=0.0,
            betaprim=-beta * drive if consistent else 0.0,
        )
    else:
        geo = replace(geo, alpha=q**2 * beta * drive * R_OVER_A if consistent else 0.0)
    cfg = replace(
        cfg,
        species=species,
        geometry=geo,
        grid=replace(
            cfg.grid, nperiod=nperiod, ntheta=32, Nz=32 * (2 * nperiod - 1), y0=1.0 / ky
        ),
        physics=replace(cfg.physics, beta=beta, use_bpar=bpar),
        terms=replace(cfg.terms, bpar=float(bpar)),
        time=replace(cfg.time, damp_ends_rate=100.0, t_max=t_max, dt=dt),
    )
    stride = max(1, int(round(0.2 / dt)))
    res = run_runtime_linear(
        cfg,
        ky_target=ky,
        Nl=Nl,
        Nm=Nm,
        solver="time",
        dt=dt,
        steps=stride * int(np.ceil(t_max / dt / stride)),
        sample_stride=stride,
        require_positive=False,
    )
    return {
        "ky": float(res.ky),
        "beta": beta,
        "tprim": tprim,
        "fprim": fprim,
        "geom": geom,
        "consistent": consistent,
        "bpar": bpar,
        "Nm": Nm,
        "nperiod": nperiod,
        "gamma": float(res.gamma),
        "omega": float(res.omega),
        "settled": bool(res.fit_settled),
    }


def _zonal_job(args: tuple) -> tuple[str, dict]:
    name, a, kw = args
    return name, {"args": a, **kw, **zonal_run(*a, **kw)}


def _em_job(kw: dict) -> dict:
    return em_run(**kw)


ZONAL_CASES = {
    "rh_q1.4": ((1.4, 0.18, 0.05), {}),
    "rh_q1.4_kx0.025": ((1.4, 0.18, 0.025), {}),
    "rh_q1.0": ((1.0, 0.18, 0.05), {}),
    "rh_q2.0": ((2.0, 0.18, 0.05), {}),
    "rh_miller_q1.4": ((1.4, 0.18, 0.05), {"model": "miller"}),
    "xc_kappa1": ((2.0, 0.2, 0.05), {"model": "miller", "kappa": 1.0}),
    "xc_kappa3": ((2.0, 0.2, 0.05), {"model": "miller", "kappa": 3.0}),
    "sw_fig1": ((1.5, 0.1, 0.131), {}),
}


def em_cases(case: str) -> list[dict]:
    if case == "pkj":
        return [
            {"ky": 0.2, "beta": b, "tprim": 6.89 / R_OVER_A, "bpar": bp}
            for bp in (False, True)
            for b in (0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018)
        ]
    if case == "az":
        return [
            {
                "ky": ky,
                "beta": 0.015,
                "tprim": rlt / R_OVER_A,
                "geom": geom,
                "consistent": True,
                "bpar": True,
                "t_max": 20.0,
                "dt": 2.5e-4,
                "nperiod": 6,
                "Nm": 32,
            }
            for geom in ("miller", "s-alpha")
            for rlt in (35.0, 15.0)
            for ky in (0.05, 0.1, 0.2)
        ]
    if case == "cht":
        return [
            {
                "ky": 0.05,
                "beta": b,
                "tprim": 6.89 / R_OVER_A,
                "geom": "miller",
                "consistent": True,
                "bpar": True,
                "t_max": 100.0,
                "nperiod": 6,
            }
            for b in (0.008, 0.010, 0.012, 0.014, 0.016)
        ]
    raise SystemExit(f"unknown case {case!r}")


def main(argv: list[str]) -> int:
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for case in argv:
            if case == "zonal":
                jobs = [(n, a, kw) for n, (a, kw) in ZONAL_CASES.items()]
                data["zonal"] = dict(pool.map(_zonal_job, jobs))
            else:
                data[case] = list(pool.map(_em_job, em_cases(case)))
            OUT.write_text(json.dumps(data, indent=1) + "\n")
    return 0


# --- from scripts/artifacts/build_landau_damping_figure.py (removed; see plan/log.md) ---
_CONVENTION = np.sqrt(2.0)

_NU_SCAN = np.array([0.020, 0.015, 0.010, 0.0075, 0.005, 0.004, 0.003, 0.002])

_TERMS = LinearTerms(
    mirror=0.0,
    curvature=0.0,
    gradb=0.0,
    diamagnetic=0.0,
    collisions=1.0,
    hypercollisions=0.0,
    end_damping=0.0,
)


def plasma_dispersion(zeta: complex) -> complex:
    """Fried-Conte ``Z(zeta)`` via the Faddeeva function."""

    return 1j * np.sqrt(np.pi) * wofz(zeta)


def exact_root(te_over_ti: float, guess: complex) -> complex:
    """Root of ``1 + T_i/T_e + zeta Z(zeta) = 0`` as ``omega/(k_par v_ti)``."""

    tau = 1.0 / te_over_ti

    def residual(v: np.ndarray) -> list[float]:
        z = complex(v[0], v[1])
        f = 1.0 + tau + z * plasma_dispersion(z)
        return [f.real, f.imag]

    sol = fsolve(residual, [guess.real, guess.imag])
    return _CONVENTION * complex(sol[0], sol[1])


def _slab_setup(hermite: int, nu: float, te_over_ti: float):
    """Build the GKX slab flux tube, cache and jitted RHS for this problem.

    ``Ly`` is deliberately huge so the single retained ``ky`` is ~1e-4: that is
    the ``k_perp rho_i -> 0`` limit the dispersion relation is written in. The
    box length in ``z`` makes the parallel harmonic satisfy ``k_par v_ti = 1``,
    so time is directly comparable with the root.
    """

    grid = build_spectral_grid(
        GridConfig(Nx=1, Ny=2, Nz=32, Lx=1.0e4, Ly=1.0e4, boundary="periodic")
    )
    geometry = SlabGeometry.from_config(
        GeometryConfig(q=1.0, s_hat=0.0, epsilon=0.0, R0=1.0)
    )
    # GKX's tau_e is T_i/T_e (docs/theory.rst), the reciprocal of the ratio this
    # script is parameterized by. T_e = T_i cannot distinguish the two, so the
    # T_e/T_i = 10 case is what actually pins the convention.
    params = linear_params_for_geometry(geometry, tau_e=1.0 / te_over_ti, nu=nu)
    cache = build_linear_cache(grid, geometry, params, 1, hermite)
    rhs = jax.jit(
        lambda state: linear_rhs_cached(
            state, cache, params, terms=_TERMS, use_jit=False
        )[0]
    )
    return grid, rhs


def evolve(
    *, hermite: int, nu: float, te_over_ti: float, t_max: float, dt: float = 0.002
) -> tuple[np.ndarray, np.ndarray]:
    """RK4-evolve GKX's linear operator; return ``(t, Re g_0)`` for the mode."""

    grid, rhs = _slab_setup(hermite, nu, te_over_ti)
    z = np.asarray(grid.z)
    weight = jnp.asarray(np.exp(-1j * z))

    state = (
        jnp.zeros(
            (1, 1, hermite, grid.ky.size, grid.kx.size, grid.z.size),
            dtype=jnp.complex128,
        )
        .at[0, 0, 0, 1, 0, :]
        .set(jnp.asarray(np.exp(1j * z)))
    )

    def project(s: jnp.ndarray) -> float:
        return float(
            jnp.real(jnp.sum(jnp.asarray(s)[0, 0, 0, 1, 0, :] * weight) / z.size)
        )

    steps = int(round(t_max / dt))
    signal = np.empty(steps + 1)
    signal[0] = project(state)
    for index in range(steps):
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * dt * k1)
        k3 = rhs(state + 0.5 * dt * k2)
        k4 = rhs(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        signal[index + 1] = project(state)

    return np.arange(steps + 1) * dt, signal


def operator_matrix(hermite: int, nu: float, te_over_ti: float) -> np.ndarray:
    """Extract GKX's linear operator for this mode as a dense Hermite matrix.

    Applies ``linear_rhs_cached`` to each Hermite basis vector, so the spectrum
    in panel (b) belongs to the production operator rather than a model of it.
    """

    grid, rhs = _slab_setup(hermite, nu, te_over_ti)
    z = np.asarray(grid.z)
    profile = jnp.asarray(np.exp(1j * z))
    weight = jnp.asarray(np.exp(-1j * z))

    base = jnp.zeros(
        (1, 1, hermite, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex128
    )
    matrix = np.zeros((hermite, hermite), dtype=complex)
    for column in range(hermite):
        response = rhs(base.at[0, 0, column, 1, 0, :].set(profile))
        matrix[:, column] = np.asarray(
            jnp.sum(jnp.asarray(response)[0, 0, :, 1, 0, :] * weight, axis=-1) / z.size
        )
    return matrix


def fit_standing_wave(
    times: np.ndarray,
    signal: np.ndarray,
    window: tuple[float, float],
    guess: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Fit ``A exp(gamma t) cos(omega t + phi)`` and return ``(gamma, omega)``."""

    mask = (times >= window[0]) & (times <= window[1])

    def model(t: np.ndarray, amp: float, gamma: float, omega: float, phase: float):
        return amp * np.exp(gamma * t) * np.cos(omega * t + phase)

    popt, _ = curve_fit(model, times[mask], signal[mask], p0=guess, maxfev=200_000)
    return float(popt[1]), float(popt[2])


# --- from scripts/artifacts/build_recurrence_closure_figure.py (removed; see plan/log.md) ---
def _hypercollision_rate(hermite: int, *, f_hyper: float = 1.0) -> np.ndarray:
    """GX Appendix-B eq. (B12) hypercollision profile, applied for ``m > 2``."""

    order = hermite / 2.0
    m = np.arange(hermite, dtype=float)
    rate = (
        np.log(10.0)
        * f_hyper
        * (order + 0.5)
        / hermite ** (order + 0.5)
        * m ** (order + 0.5)
    )
    return np.where(m > 2, rate, 0.0)


def free_streaming_revival(
    hermite: int, closure: str, *, t_max: float | None = None, dt: float = 0.002
) -> tuple[np.ndarray, np.ndarray]:
    """Evolve the free-streaming Hermite hierarchy from ``g_0 = 1``.

    ``k_par v_ti = 1``, so the recurrence time is ``2 sqrt(N_m)`` directly.
    """

    limit = t_max if t_max is not None else 3.2 * np.sqrt(hermite)
    m = np.arange(hermite, dtype=float)
    up = np.sqrt(m + 1.0)
    down = np.sqrt(m)

    damping = np.zeros(hermite)
    if closure == "hypercollisions":
        damping = _hypercollision_rate(hermite)
    tail = 0.0
    if closure == "reflectionless":
        tail = hermite_closure_coefficient(hermite) * np.sqrt(hermite)

    def rhs(state: np.ndarray) -> np.ndarray:
        out = np.zeros_like(state)
        out[:-1] += -1j * up[:-1] * state[1:]
        out[1:] += -1j * down[1:] * state[:-1]
        out -= damping * state
        if tail:
            # -R sqrt(M+1) v |k_par| G_M, confined to the last moment.
            out[-1] -= tail * state[-1]
        return out

    state = np.zeros(hermite, dtype=complex)
    state[0] = 1.0
    steps = int(round(limit / dt))
    times = np.arange(steps + 1) * dt
    amplitude = np.empty(steps + 1)
    amplitude[0] = 1.0
    for index in range(steps):
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * dt * k1)
        k3 = rhs(state + 0.5 * dt * k2)
        k4 = rhs(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        amplitude[index + 1] = abs(state[0])
    return times, amplitude


def measure_revival(hermite: int, closure: str) -> tuple[float, float]:
    """Peak ``|g_0|`` in the recurrence window, and the time it occurs."""

    times, amplitude = free_streaming_revival(hermite, closure)
    predicted = 2.0 * np.sqrt(hermite)
    window = (times > 0.6 * predicted) & (times < 1.9 * predicted)
    peak = float(amplitude[window].max())
    when = float(times[window][np.argmax(amplitude[window])])
    return peak, when


def resolved_window_error(hermite: int, closure: str) -> float:
    """Max error in ``g_0`` before recurrence, against a converged reference.

    This is the metric that makes the comparison fair. Revival amplitude alone
    rewards any strong damping; a closure that flattens the whole hierarchy
    scores perfectly on it. Hypercollisions act on a BAND of moments (m > 2),
    so they can suppress the revival while also perturbing resolved physics,
    whereas the reflectionless condition touches only m = M. Measuring the
    resolved window against a reference charges for that.
    """

    horizon = 2.0 * np.sqrt(hermite) + 1.0
    _, reference = free_streaming_revival(1024, "truncation", t_max=horizon)
    _, trial = free_streaming_revival(hermite, closure, t_max=horizon)
    count = min(reference.size, trial.size)
    return float(np.abs(trial[:count] - reference[:count]).max())


# --- from scripts/comparison/compare_gx_rhs_terms.py (removed; see plan/log.md) ---
REFERENCE_NU_HYPER_L = 0.0

REFERENCE_NU_HYPER_M = 1.0

REFERENCE_P_HYPER_L = 6.0

REFERENCE_P_HYPER_M = 20.0


def _reference_hypercollision_power(nhermite: int | None) -> float:
    if nhermite is None:
        return REFERENCE_P_HYPER_M
    return float(min(REFERENCE_P_HYPER_M, max(int(nhermite) // 2, 1)))


def _apply_reference_hypercollisions(
    params: LinearParams, *, nhermite: int | None = None
) -> LinearParams:
    return replace(
        params,
        nu_hyper=0.0,
        nu_hyper_l=REFERENCE_NU_HYPER_L,
        nu_hyper_m=REFERENCE_NU_HYPER_M,
        p_hyper_l=REFERENCE_P_HYPER_L,
        p_hyper_m=_reference_hypercollision_power(nhermite),
        hypercollisions_const=0.0,
        hypercollisions_kz=1.0,
    )


def _two_species_params(
    model,
    *,
    kpar_scale: float,
    omega_d_scale: float,
    omega_star_scale: float,
    rho_star: float,
    beta_override: float | None = None,
    fapar_override: float | None = None,
    apar_beta_scale: float | None = None,
    ampere_g0_scale: float | None = None,
    bpar_beta_scale: float | None = None,
    damp_ends_amp: float | None = None,
    damp_ends_widthfrac: float | None = None,
    nhermite: int | None = None,
) -> LinearParams:
    """Build ``LinearParams`` for a two-species kinetic model."""

    mass_ratio = float(model.mass_ratio)
    if mass_ratio <= 0.0:
        raise ValueError("mass_ratio must be > 0")
    Te_over_Ti = float(model.Te_over_Ti)
    if Te_over_Ti <= 0.0:
        raise ValueError("Te_over_Ti must be > 0")
    ion_fprim_raw = getattr(model, "fprim_i", None)
    ele_fprim_raw = getattr(model, "fprim_e", None)
    ion_fprim = float(model.fprim) if ion_fprim_raw is None else float(ion_fprim_raw)
    ele_fprim = float(model.fprim) if ele_fprim_raw is None else float(ele_fprim_raw)

    nu_i = float(getattr(model, "nu_i", 0.0))
    nu_e = float(getattr(model, "nu_e", 0.0))
    beta = float(getattr(model, "beta", 1.0e-5))
    if beta_override is not None:
        beta = float(beta_override)

    ion = Species(
        charge=1.0,
        mass=1.0,
        density=1.0,
        temperature=1.0,
        tprim=float(getattr(model, "tprim_i", model.tprim_e)),
        fprim=ion_fprim,
        nu=nu_i,
    )
    electron = Species(
        charge=-1.0,
        mass=1.0 / mass_ratio,
        density=1.0,
        temperature=Te_over_Ti,
        tprim=float(model.tprim_e),
        fprim=ele_fprim,
        nu=nu_e,
    )
    params = build_linear_params(
        [ion, electron],
        tau_e=0.0,
        kpar_scale=kpar_scale,
        omega_d_scale=omega_d_scale,
        omega_star_scale=omega_star_scale,
        rho_star=rho_star,
        beta=beta,
        fapar=1.0 if beta > 0.0 else 0.0,
        apar_beta_scale=0.5 if apar_beta_scale is None else float(apar_beta_scale),
        ampere_g0_scale=0.5 if ampere_g0_scale is None else float(ampere_g0_scale),
        bpar_beta_scale=0.5 if bpar_beta_scale is None else float(bpar_beta_scale),
    )
    params = _apply_reference_hypercollisions(params, nhermite=nhermite)
    if fapar_override is not None:
        params = replace(params, fapar=float(fapar_override))
    if damp_ends_amp is not None:
        params = replace(params, damp_ends_amp=float(damp_ends_amp))
    if damp_ends_widthfrac is not None:
        params = replace(params, damp_ends_widthfrac=float(damp_ends_widthfrac))
    return params


def _build_gaussian_profile(
    z: np.ndarray,
    *,
    kx: float,
    ky: float,
    s_hat: float,
    init_cfg: InitializationConfig,
) -> np.ndarray:
    if ky == 0.0:
        return np.zeros_like(z)
    theta0 = kx / (s_hat * ky)
    envelope = (
        init_cfg.gaussian_envelope_constant
        + init_cfg.gaussian_envelope_sine * np.sin(z - theta0)
    )
    width = init_cfg.gaussian_width
    if width <= 0.0:
        raise ValueError("gaussian_width must be > 0")
    return envelope * np.exp(-(((z - theta0) / width) ** 2))


def _build_initial_condition(
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    *,
    ky_index: int | Sequence[int] | np.ndarray,
    kx_index: int,
    Nl: int,
    Nm: int,
    init_cfg: InitializationConfig,
) -> jnp.ndarray:
    init_field = init_cfg.init_field.lower()
    field_map = {
        "density": (0, 0),
        "upar": (0, 1),
        "tpar": (0, 2),
        "tperp": (1, 0),
        "qpar": (0, 3),
        "qperp": (1, 1),
    }
    # Moment-normalized initializer amplitudes for init_field="all".
    all_scales = {
        "density": 1.0,
        "upar": 1.0,
        "tpar": 1.0 / np.sqrt(2.0),
        "tperp": 1.0,
        "qpar": 1.0 / np.sqrt(6.0),
        "qperp": 1.0,
    }
    if init_field != "all" and init_field not in field_map:
        raise ValueError(
            "init_field must be one of {'density','upar','tpar','tperp','qpar','qperp','all'}"
        )

    G0 = np.zeros((Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
    amp = float(init_cfg.init_amp)
    ky_idx = np.atleast_1d(np.asarray(ky_index, dtype=int))
    for ky_i in ky_idx:
        if init_cfg.gaussian_init:
            profile = _build_gaussian_profile(
                np.asarray(grid.z),
                kx=float(grid.kx[kx_index]),
                ky=float(grid.ky[ky_i]),
                s_hat=geom.s_hat,
                init_cfg=init_cfg,
            )
            init_vals = amp * profile * (1.0 + 1.0j)
        else:
            init_vals = amp * (1.0 + 1.0j) * np.ones_like(grid.z)
        if grid.ky[ky_i] != 0.0:
            if init_field == "all":
                for field_name, (l_idx, m_idx) in field_map.items():
                    if l_idx < Nl and m_idx < Nm:
                        scale = all_scales.get(field_name, 1.0)
                        G0[l_idx, m_idx, ky_i, kx_index, :] = init_vals * scale
            else:
                l_idx, m_idx = field_map[init_field]
                if l_idx >= Nl or m_idx >= Nm:
                    raise ValueError("init_field moment exceeds (Nl, Nm) resolution")
                G0[l_idx, m_idx, ky_i, kx_index, :] = init_vals
    return jnp.asarray(G0)
