"""Physics gate: GKX must reproduce the exact kinetic Landau roots.

The reference is the slab gyrokinetic ion-acoustic dispersion relation with
adiabatic electrons at ``k_perp -> 0``,

    1 + T_i/T_e + zeta Z(zeta) = 0,

solved here from ``scipy.special.wofz`` rather than read from a table, so the
gate cannot drift with a hard-coded constant.

Three separate traps make a naive version of this test pass while measuring the
wrong thing, and each is gated explicitly below:

* a collisionless truncated Hermite system has a purely real spectrum, so it has
  no asymptotic damping to measure at all;
* the Landau root is not an eigenvalue of the collisional operator either -- it
  is a pole of the continued response, reached by ``nu -> 0`` extrapolation;
* a density perturbation with no initial flow is a standing wave, so its phase
  does not advance and ``omega`` must come from an envelope-and-oscillation fit.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tools.artifacts.build_landau_damping_figure import (
    _NU_SCAN,
    evolve,
    exact_root,
    fit_standing_wave,
    operator_matrix,
)

pytestmark = pytest.mark.skipif(
    jnp.zeros(1).dtype != jnp.float64,
    reason="Landau roots need float64; CI runs JAX_ENABLE_X64",
)


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
