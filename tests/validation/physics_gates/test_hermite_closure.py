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
    from tools.artifacts.build_recurrence_closure_figure import free_streaming_revival

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
    from tools.artifacts.build_recurrence_closure_figure import (
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
