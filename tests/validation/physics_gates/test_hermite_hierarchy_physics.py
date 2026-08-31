"""Physics gates: the parallel-velocity (Hermite) hierarchy.

One truncated hierarchy underlies every gate here: free streaming pushes free
energy up in m, a hard truncation reflects it back as recurrence, and what
the closure does with that pulse decides whether the code can report Landau
damping, a recurrence time, or a stable design at all. The blocks below keep
the module docstring of the file they came from, because those docstrings
record the provenance of the reference values being asserted.
"""

from __future__ import annotations

import dataclasses
import math

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest

from gkx.artifacts.figure_style import save_figure
from gkx.geometry.analytic import SAlphaGeometry
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
from tools.artifacts.build_landau_damping_figure import (
    _NU_SCAN,
    evolve,
    exact_root,
    fit_standing_wave,
    operator_matrix,
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
