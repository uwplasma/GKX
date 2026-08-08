"""Physics: can the linear objective report that a design is stable?

An optimizer follows the objective, so an objective with a floor at zero is not
merely imprecise -- it can never say "this design is quiet", and wherever the
physical branch is weak it returns whatever marginal mode happens to sit highest.

That is what the shipped defaults did. Every dissipation amplitude in
``_default_gradient_linear_params`` was zero, which leaves a truncated Hermite
hierarchy with no velocity-space dissipation, so at zero drive the whole spectrum
sits on the imaginary axis and ``argmax(Re lambda)`` is floored at zero or above.
Measured on Cyclone at zero drive: ``+9.3e-05`` at ``N_m = 8``, ``+6.4e-14`` at
16, ``+1.0e-13`` at 32, at ``|omega|`` between 20 and 90.

Zero drive is the sharpest available check because it needs no reference value.
With ``tprim = fprim = 0`` there is no free energy in the system, so every mode
must be damped, and any positive growth rate is numerical by construction.

These would fail if the closure were removed from the defaults, and the
convergence case would fail if it were replaced by hyperdiffusion, which in this
path is exactly ``-D_hyper * I`` and so shifts every eigenvalue by a constant
without restoring resolution.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import pytest

from gkx.geometry.analytic import SAlphaGeometry
from gkx.geometry.flux_tube import sample_flux_tube_geometry
from gkx.objectives.core import (
    _default_gradient_linear_params,
    solver_objective_vector_from_geometry,
)

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
