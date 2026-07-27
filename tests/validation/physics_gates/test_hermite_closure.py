"""Physics gate: the reflectionless Hermite closure.

Closing the hierarchy with ``g_{M+1} = 0`` makes the truncation a reflecting
wall. The free-energy pulse streams up in ``m``, hits the wall, and returns as
recurrence. Measured on the free-streaming hierarchy, the revival of ``|g_0|``
after ``t_rec ~ 2 sqrt(M)`` reaches 8-16 times the *initial* amplitude, so the
signal after recurrence is not physics at all.

The outgoing condition of Kanekar, Schekochihin, Dorland & Loureiro,
*J. Plasma Phys.* 81, 305810104 (2015), equation (4.36) absorbs the pulse
instead. It costs one extra term on the last moment, has no free parameter, and
becomes exact as the resolution grows.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from gkx.operators.linear.streaming import abs_z_periodic
from gkx.terms.linear_terms import (
    hermite_closure_coefficient,
    linked_streaming_contribution,
)


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
    assert abs(
        hermite_closure_coefficient(3) - float(np.sqrt(8.0 / np.pi) / np.sqrt(3.0))
    ) < 1.0e-14


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
    expected = -coefficient * math.sqrt(hermite_count) * np.asarray(
        abs_z_periodic(state[:, :, hermite_count - 1], kz=wavenumbers)
    )
    # The matrices are assembled in the ambient precision, so the agreement
    # floor follows it: ~1e-13 under x64, ~1e-4 under the float32 policy.
    tolerance = 1.0e-10 if jnp.zeros(1).dtype == jnp.float64 else 1.0e-3
    scale = max(float(np.abs(expected[0, 0]).max()), 1.0)
    assert np.abs(difference[0, 0, -1] - expected[0, 0]).max() < tolerance * scale

    # Strictly dissipative: it can only remove free energy from the last moment.
    production = float(
        np.real(np.sum(np.conj(state[0, 0, -1]) * difference[0, 0, -1]))
    )
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
