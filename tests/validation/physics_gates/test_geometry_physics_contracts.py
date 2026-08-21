"""Physics gate: geometry contracts that silently corrupt results when broken.

Both properties gated here were wrong in ways that produce plausible-looking
numbers rather than errors, which is why they are worth a dedicated gate.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from gkx.config import GeometryConfig, GridConfig
from gkx.core.grid import build_spectral_grid
from gkx.geometry import SAlphaGeometry, SlabGeometry
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import (
    LinearParams,
    LinearTerms,
    linear_params_for_geometry,
)
from gkx.operators.linear.rhs import linear_rhs_cached


def test_s_alpha_keeps_radial_wavenumber_at_zero_shear() -> None:
    r"""``k_perp^2`` must depend on ``kx`` even when the magnetic shear vanishes.

    ``k_perp2`` divides ``kx`` by ``s_hat`` and multiplies by ``gds22``, so the
    two cancel at finite shear. At zero shear it uses ``kx`` directly, which
    means ``gds22`` has to be 1 rather than ``s_hat**2``. Leaving it at zero
    erased the ``kx`` dependence entirely: every radial mode then shares one
    perpendicular wavenumber, hence identical FLR factors and a degenerate
    zonal polarization. ``SlabGeometry`` always carried this guard.
    """

    kx = jnp.asarray([0.0, 0.3, 0.6])
    ky = jnp.asarray(0.5)
    theta = jnp.asarray(0.0)

    ratios = []
    for shear in (0.8, 0.4, 1.0e-6, 0.0):
        geometry = SAlphaGeometry.from_config(
            GeometryConfig(q=1.4, s_hat=shear, epsilon=0.18, R0=2.78)
        )
        kperp2 = np.asarray(geometry.k_perp2(kx, ky, theta))
        assert np.all(np.diff(kperp2) > 0.0), (
            f"s_hat={shear}: k_perp2 does not increase with kx: {kperp2}"
        )
        ratios.append(kperp2[2] / kperp2[0])

    # At theta = 0 the shear terms drop out of k_perp2, so the kx dependence
    # must be identical at every shear. That continuity is the sharpest
    # statement: the zero-shear branch cannot be special-cased incorrectly.
    assert max(ratios) - min(ratios) < 1.0e-9, f"kx scaling varies with shear: {ratios}"

    slab = SlabGeometry.from_config(
        GeometryConfig(q=1.4, s_hat=0.0, epsilon=0.18, R0=2.78)
    )
    assert np.all(np.diff(np.asarray(slab.k_perp2(kx, ky, theta))) > 0.0)


def test_s_alpha_retains_field_strength_variation() -> None:
    """s-alpha must keep ``B(theta)``, which is what makes particles trap.

    Trapping supplies the neoclassical polarization behind the
    Rosenbluth-Hinton residual, so a uniform ``|B|`` would silently remove that
    physics while every other term kept working.
    """

    theta = jnp.linspace(-np.pi, np.pi, 33)
    for epsilon in (0.1, 0.18):
        geometry = SAlphaGeometry.from_config(
            GeometryConfig(q=1.4, s_hat=0.8, epsilon=epsilon, R0=2.78)
        )
        bmag = np.asarray(geometry.bmag(theta))
        # B = B0 / (1 + eps cos theta)
        assert bmag.min() == np.float64(bmag.min())
        assert abs(bmag.min() - 1.0 / (1.0 + epsilon)) < 1.0e-9
        assert abs(bmag.max() - 1.0 / (1.0 - epsilon)) < 1.0e-9
        assert np.abs(np.asarray(geometry.bgrad(theta))).max() > 0.0


def _streaming_only_rhs(geometry, params) -> float:
    """Return the norm of a streaming-only linear RHS for a z-varying state."""

    grid = build_spectral_grid(GridConfig(Nx=3, Ny=2, Nz=16, Lx=20.0, Ly=20.0))
    cache = build_linear_cache(grid, geometry, params, 4, 4)
    state = jnp.zeros(
        (4, 4, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex128
    )
    # A z-dependent perturbation: a constant one has zero parallel derivative
    # and would make this test vacuous.
    state = state.at[0, 0, 1, 1, :].set(jnp.sin(jnp.asarray(grid.z)) * 1.0e-3)
    terms = LinearTerms(
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
    )
    return float(
        jnp.linalg.norm(
            linear_rhs_cached(state, cache, params, terms=terms, use_jit=False)[0]
        )
    )


def test_geometry_params_helper_carries_the_parallel_scale() -> None:
    """``linear_params_for_geometry`` must make streaming follow ``gradpar``.

    ``kpar_scale`` multiplies the parallel derivative and defaults to 1.0,
    because ``LinearParams`` knows nothing about geometry. A hand-built
    ``LinearParams()`` therefore streams at the wrong rate -- by ``1/(q R0)``,
    a factor of 8.4 for the q=2.8, R0=3 case below -- with no error raised.
    The runtime path sets it from the geometry; this helper is the Python-API
    equivalent, and this test pins that it actually takes effect.
    """

    baseline = None
    for q, major_radius in ((1.4, 1.0), (1.4, 2.78), (2.8, 3.0)):
        geometry = SAlphaGeometry.from_config(
            GeometryConfig(q=q, s_hat=0.8, epsilon=0.18, R0=major_radius)
        )
        gradpar = float(np.asarray(geometry.gradpar()))

        plain = _streaming_only_rhs(geometry, LinearParams())
        matched = _streaming_only_rhs(geometry, linear_params_for_geometry(geometry))

        # The bare constructor ignores geometry entirely, so it gives the same
        # answer for every case; that is the defect this helper exists for.
        if baseline is None:
            baseline = plain
        assert abs(plain - baseline) < 1.0e-15 * baseline

        # The helper scales the parallel term by exactly gradpar.
        assert abs(matched / plain - gradpar) < 1.0e-9, (
            f"q={q}, R0={major_radius}: streaming scaled by {matched / plain:.6f}, "
            f"expected gradpar {gradpar:.6f}"
        )

    # An explicit override still wins, for published normalizations that fold
    # q R0 in elsewhere.
    geometry = SAlphaGeometry.from_config(
        GeometryConfig(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.78)
    )
    assert linear_params_for_geometry(geometry, kpar_scale=1.0).kpar_scale == 1.0


def test_electromagnetic_zonal_solve_is_continuous_in_beta() -> None:
    """The field solve must not jump when beta becomes infinitesimally finite.

    The adiabatic species responds to ``phi - <phi>``, so quasineutrality
    carries a ``tau_e<phi>`` source at ``ky = 0``. The electrostatic branch
    solves for it exactly; the electromagnetic branch used to rebuild ``phi``
    from ``nbar`` alone and drop it, which over-screened the zonal potential by
    a factor of 7.3 and made the solve discontinuous as ``beta -> 0``.

    Continuity in ``beta`` is the sharp statement: no physical quantity may
    jump between ``beta = 0`` and ``beta = 1e-12``.
    """

    from gkx.terms.fields import solve_fields

    grid = build_spectral_grid(GridConfig(Nx=5, Ny=2, Nz=16, Lx=40.0, Ly=40.0))
    geometry = SAlphaGeometry.from_config(
        GeometryConfig(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.78)
    )
    generator = np.random.default_rng(0)
    shape = (1, 4, 4, grid.ky.size, grid.kx.size, grid.z.size)
    state = jnp.asarray(
        generator.normal(size=shape) * 1.0e-3
        + 1j * generator.normal(size=shape) * 1.0e-3
    )
    unit = jnp.asarray([1.0])

    def zonal_potential(beta: float) -> np.ndarray:
        params = linear_params_for_geometry(geometry, beta=beta, tau_e=1.0)
        cache = build_linear_cache(grid, geometry, params, 4, 4)
        fields = solve_fields(
            state,
            cache,
            params,
            charge=unit,
            density=unit,
            temp=unit,
            mass=unit,
            tz=unit,
            vth=unit,
            fapar=jnp.asarray(0.0),
            w_bpar=jnp.asarray(1.0),
        )
        return np.asarray(fields.phi)

    electrostatic = zonal_potential(0.0)
    infinitesimal = zonal_potential(1.0e-12)

    zonal_es = np.abs(electrostatic[..., 0, 1, :]).max()
    zonal_em = np.abs(infinitesimal[..., 0, 1, :]).max()
    assert zonal_es > 0.0
    assert abs(zonal_em / zonal_es - 1.0) < 1.0e-9, (
        f"zonal potential jumps by {zonal_em / zonal_es:.4f} between beta=0 and "
        "beta=1e-12; the ky=0 adiabatic correction is missing from the "
        "electromagnetic branch"
    )

    # Finite beta must still do something physical rather than nothing.
    assert np.abs(zonal_potential(1.0e-2)[..., 0, 1, :]).max() / zonal_es < 0.99


def test_reference_electrostatic_solve_matches_production_including_zonal() -> None:
    """The reference field solve must reproduce production at ``ky = 0`` too.

    The existing equivalence test builds its fixture with ``Nx = 1``, which
    makes the ``kx > 0`` mask empty, so the zonal branch is never exercised and
    both implementations trivially return ``nbar/q_phi``. With ``Nx > 1`` the
    reference path returned 0.22 times the production zonal potential while
    agreeing exactly everywhere else.
    """

    from gkx.parallel.velocity_drive import electrostatic_phi_reference
    from gkx.terms.fields import solve_fields

    grid = build_spectral_grid(
        GridConfig(Nx=5, Ny=2, Nz=16, Lx=40.0, Ly=40.0, boundary="periodic")
    )
    geometry = SAlphaGeometry.from_config(
        GeometryConfig(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.78)
    )
    assert grid.kx.size > 1, "the fixture must expose a nonzero kx to be meaningful"

    generator = np.random.default_rng(1)
    shape = (1, 4, 4, grid.ky.size, grid.kx.size, grid.z.size)
    state = jnp.asarray(
        generator.normal(size=shape) * 1.0e-3
        + 1j * generator.normal(size=shape) * 1.0e-3
    )
    unit = jnp.asarray([1.0])
    params = linear_params_for_geometry(geometry, tau_e=1.0)
    cache = build_linear_cache(grid, geometry, params, 4, 4)

    production = np.asarray(
        solve_fields(
            state,
            cache,
            params,
            charge=unit,
            density=unit,
            temp=unit,
            mass=unit,
            tz=unit,
            vth=unit,
            fapar=jnp.asarray(0.0),
            w_bpar=jnp.asarray(0.0),
        ).phi
    )
    reference = np.asarray(
        electrostatic_phi_reference(
            state,
            Jl=cache.Jl,
            tau_e=params.tau_e,
            charge=unit,
            density=unit,
            tz=unit,
            mask0=cache.mask0,
            jacobian=cache.jacobian,
            ky=cache.ky,
        )
    )

    zonal = np.abs(production[..., 0, 1, :]).max()
    assert zonal > 0.0, "the zonal mode must carry signal for this test to bite"
    assert np.abs(reference - production).max() < 1.0e-12 * np.abs(production).max()
