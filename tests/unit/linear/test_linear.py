"""Linear operator tests for the flux-tube electrostatic model."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gkx.config import CycloneBaseCase, GridConfig, GeometryConfig
from scripts.checks._gates.validation_gates import (
    estimate_observed_order,
)
from gkx.geometry import SAlphaGeometry, SlabGeometry, sample_flux_tube_geometry
from gkx.core_grid import build_spectral_grid, select_ky_grid
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.cache_model import LinearCache
from gkx.operators.linear.moments import (
    apply_hermite_v,
    apply_laguerre_x,
    build_H,
    compute_b,
    diamagnetic_drive_coeffs,
    energy_operator,
    grad_z_periodic,
    quasineutrality_phi,
    streaming_term,
)
from gkx.operators.linear.params import LinearParams, LinearTerms
from gkx.operators.linear.rhs import linear_rhs, linear_rhs_cached
from gkx.solvers_linear_integrators import (
    integrate_linear,
    integrate_linear_diagnostics,
)
from gkx.operators.linear.linked import _build_linked_fft_maps
from gkx.operators.linear.params import _x64_enabled
from gkx.operators.linear.streaming import grad_z_linked_fft
from gkx.core_velocity import J_l_all
from gkx.solvers_linear_krylov import dominant_eigenpair
from gkx.solvers_linear_implicit import _build_implicit_operator
from gkx.terms.linear_terms import (
    collision_invariant_rates,
    collision_quadratic_rate,
    collisions_contribution,
    conservative_full_f_dougherty_cross_moments,
    drift_kinetic_coulomb_six_moment_contribution,
    drift_kinetic_sugama_six_moment_contribution,
    multispecies_collision_invariant_rates,
)
from scripts.checks._gates.validation_gates import (
    drift_kinetic_dougherty_contribution,
)
from gkx.terms.assembly import assemble_rhs_terms_cached
import inspect
from types import SimpleNamespace
from gkx.geometry import FluxTubeGeometryData
from gkx.core_velocity import _gyro_bessel_factors
import gkx.operators.linear as linear_cache
import gkx.solvers_linear_integrators as linear_integrators
import gkx.solvers_linear_integrators as linear_diagnostics
import gkx.solvers_linear_implicit as linear_implicit
import gkx.solvers_linear_krylov_algorithms as krylov_algorithms
import gkx.operators.linear.dissipation as linear_dissipation
import gkx.terms.linear_terms as linear_terms
from gkx.operators.linear.cache_arrays import (
    hypercollision_damping,
    _build_end_damping_profile_array,
    _build_gyroaverage_cache_arrays,
    _build_low_rank_moment_cache_arrays,
)
from gkx.operators.linear.cache_builder import (
    linked_chain_cover_mask,
    mask_off_chain_rows,
)
from gkx.operators.linear.linked import (
    linked_cover_mask_from_cache,
    mask_supplied_state,
    _build_linked_end_damping_profile,
    _signed_to_index,
)
from gkx.operators.linear.moments import lenard_bernstein_eigenvalues
from gkx.solvers_linear_krylov_algorithms import _linked_covered_mode_mask
from gkx.operators.linear.params import (
    linear_terms_to_term_config,
    term_config_to_linear_terms,
    _SPECIES_PARAM_NAMES,
    _as_species_array,
    _check_nonnegative,
    _check_positive,
    _is_tracer,
    _resolve_implicit_preconditioner,
)
from gkx.solvers_linear_integrators import (
    _integrate_linear_cached_impl,
)
from gkx.solvers_linear_parallel import (
    linear_rhs_parallel_cached,
    _is_electrostatic_field_terms,
    _is_electrostatic_slice_terms,
    _is_streaming_only_terms,
    _resolve_parallel_devices,
)
from gkx.solvers_linear_implicit import (
    _integrate_linear_implicit_cached,
)
from gkx.terms.config import FieldState, TermConfig
from gkx.core_velocity import hermite_ladder_coeffs
import gkx.operators.linear.moments as linear_moments
import gkx.operators.linear.streaming as streaming_operators
import dataclasses
import tomllib
import warnings
from support.paths import REPO_ROOT
from support.paths import load_tool_script


def test_grad_z_periodic_sine():
    """Centered periodic derivative should differentiate a sine wave."""
    z = jnp.linspace(0.0, 2.0 * jnp.pi, 64, endpoint=False)
    dz = z[1] - z[0]
    f = jnp.sin(z)
    df = grad_z_periodic(f, dz)
    assert jnp.allclose(df, jnp.cos(z), atol=2.0e-2)


@pytest.mark.parametrize("nz", [31, 32])
def test_build_linked_fft_maps_keeps_real_fft_positive_ky_modes(nz):
    kx = np.array([0.0], dtype=float)
    ky = np.array([0.0, 0.01, 0.02], dtype=float)
    linked_indices, linked_kz = _build_linked_fft_maps(
        kx=kx,
        ky=ky,
        y0=100.0,
        jtwist=2,
        dz=(2.0 * np.pi) / 32.0,
        nz=nz,
        real_dtype=jnp.float32,
        ky_mode=np.array([0, 1, 2], dtype=int),
    )

    assert len(linked_indices) == 1
    assert np.array_equal(
        np.asarray(linked_indices[0]), np.array([[0], [1], [2]], dtype=np.int32)
    )
    modes = np.r_[np.arange((nz + 1) // 2), np.arange(-(nz // 2), 0)]
    np.testing.assert_allclose(linked_kz[0], modes * 32 / nz, rtol=1e-6, atol=1e-6)


def test_build_linear_cache_zero_shat_periodic_uses_periodic_fft_without_end_damping():
    from gkx.geometry import SlabGeometry, apply_geometry_grid_defaults
    from gkx.config import GeometryConfig
    from gkx.core_grid import select_real_fft_ky_grid
    from gkx.operators.linear.params import Species, build_linear_params

    geom = SlabGeometry.from_config(
        GeometryConfig(model="slab", s_hat=1.0e-8, zero_shat=True)
    )
    grid_cfg = apply_geometry_grid_defaults(
        geom,
        GridConfig(
            Nx=1,
            Ny=7,
            Nz=32,
            Lx=2.0 * np.pi,
            Ly=200.0 * np.pi,
            boundary="linked",
            y0=100.0,
        ),
    )
    grid_full = build_spectral_grid(grid_cfg)
    grid = select_real_fft_ky_grid(
        grid_full, np.array([0.0, 0.01, 0.02], dtype=np.float32)
    )
    params = build_linear_params(
        (
            Species(
                charge=1.0, mass=1.0, density=1.0, temperature=1.0, tprim=0.0, fprim=0.0
            ),
        ),
        tau_e=0.0,
        kpar_scale=float(geom.gradpar()),
        beta=0.01,
        fapar=1.0,
    )

    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=4)

    assert cache.use_twist_shift is False
    assert cache.jtwist == 0
    assert len(cache.linked_indices) == 0
    assert np.allclose(np.asarray(cache.damp_profile), 0.0)
    assert cache.linked_damp_profile.size == 0


def test_build_linear_cache_single_selected_ky_ignores_nonlinear_dealias_mask():
    grid_cfg = GridConfig(
        Nx=1, Ny=16, Nz=32, Lx=2.0 * np.pi, Ly=0.628, boundary="periodic", y0=0.2
    )
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid_full = build_spectral_grid(cfg.grid)
    grid = select_ky_grid(
        grid_full, 6
    )  # ky = 30 on this grid; masked by 2/3 in the full nonlinear mesh
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()

    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=2)

    assert np.asarray(grid_full.dealias_mask)[6, 0].item() is False
    assert np.asarray(grid.dealias_mask).item() is True
    assert float(np.nanmax(np.asarray(cache.kperp2))) > 0.0


def test_build_linear_cache_multi_selected_ky_ignores_nonlinear_dealias_mask():
    grid_cfg = GridConfig(
        Nx=1, Ny=16, Nz=32, Lx=2.0 * np.pi, Ly=0.628, boundary="periodic", y0=0.2
    )
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid_full = build_spectral_grid(cfg.grid)
    grid = select_ky_grid(grid_full, [5, 6])
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()

    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=2)

    assert not bool(np.asarray(grid_full.dealias_mask)[6, 0])
    assert np.all(np.asarray(grid.dealias_mask))
    assert float(np.nanmin(np.asarray(cache.kperp2[:, 0, :]))) > 0.0


def test_linked_fft_derivative_matches_periodic_for_one_link_chains():
    rng = np.random.default_rng(0)
    f = (rng.normal(size=(2, 3, 1, 32)) + 1j * rng.normal(size=(2, 3, 1, 32))).astype(
        np.complex64
    )
    dz = (2.0 * np.pi) / 32.0
    linked_indices, linked_kz = _build_linked_fft_maps(
        kx=np.array([0.0], dtype=float),
        ky=np.array([0.0, 0.01, 0.02], dtype=float),
        y0=100.0,
        jtwist=2,
        dz=dz,
        nz=32,
        real_dtype=jnp.float32,
        ky_mode=np.array([0, 1, 2], dtype=int),
    )

    df_periodic = grad_z_periodic(jnp.asarray(f), dz=dz)
    df_linked = grad_z_linked_fft(
        jnp.asarray(f),
        dz=dz,
        linked_indices=linked_indices,
        linked_kz=linked_kz,
        linked_full_cover=True,
        linked_inverse_permutation=jnp.arange(3, dtype=jnp.int32),
        linked_use_gather=True,
        linked_gather_map=jnp.arange(3, dtype=jnp.int32),
        linked_gather_mask=jnp.ones(3, dtype=bool),
    )

    assert jnp.allclose(df_linked, df_periodic, rtol=1.0e-6, atol=2.0e-6)


def test_compute_b_shape_and_value(cyclone_world):
    """b should match k_perp^2 for s-alpha geometry."""
    cfg, grid, geom = cyclone_world(Nx=8, Ny=6, Nz=8, Lx=6.0, Ly=6.0)
    b = compute_b(grid, geom, rho=1.0)
    assert b.shape == (grid.ky.size, cfg.grid.Nx, cfg.grid.Nz)
    kx0 = grid.kx[0]
    ky0 = grid.ky[0]
    theta0 = grid.z[0]
    kx_eff = kx0 + geom.s_hat * ky0 * theta0
    assert jnp.isclose(b[0, 0, 0], kx_eff * kx_eff + ky0 * ky0)


def test_slab_itg_matrix_matches_published_gyro_moment_hierarchy(only_terms) -> None:
    r"""The runtime must reproduce Frei et al. (2022), equations (2.14)--(2.18)."""

    maximum_hermite, maximum_laguerre = 3, 2
    nm, nl = maximum_hermite + 1, maximum_laguerre + 1
    grid = build_spectral_grid(
        GridConfig(Nx=1, Ny=4, Nz=8, Lx=2.0 * np.pi, Ly=4.0 * np.pi)
    )
    params = LinearParams(
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
        grid, SlabGeometry(s_hat=0.0, z0=10.0), params, Nl=nl, Nm=nm
    )
    terms = only_terms(streaming=1.0, diamagnetic=1.0)
    phase = np.exp(1j * np.asarray(grid.z))
    mode_count = nl * nm
    observed = np.empty((mode_count, mode_count), dtype=complex)
    for column in range(mode_count):
        laguerre_order, hermite_order = divmod(column, nm)
        # Row 1 is ``+ky_1`` on either ky layout, and the published matrix is a
        # single-mode statement, so the axis length is the grid's to state and
        # the row index is not.
        state = np.zeros((1, nl, nm, int(grid.ky.size), 1, 8), dtype=complex)
        state[0, laguerre_order, hermite_order, 1, 0] = phase
        rhs, _ = linear_rhs_cached(
            jnp.asarray(state), cache, params, terms=terms, use_jit=False
        )
        observed[:, column] = np.asarray(rhs)[0, :, :, 1, 0, 0].reshape(-1) / phase[0]

    kperp, kpar, tau, eta = 0.5, 0.1, 1.0, 3.0
    bessel_argument = np.sqrt(2.0 * tau) * kperp
    argument = 0.25 * bessel_argument**2
    kernels = [np.exp(-argument)]
    for order in range(1, maximum_laguerre + 2):
        kernels.append(kernels[-1] * argument / order)
    kernels = np.asarray(kernels)
    denominator = 1.0 + (1.0 - np.sum(kernels[:nl] ** 2)) / tau

    # Assemble the paper's N_(p,j) ordering before converting to the runtime's
    # signed-Laguerre G_(j,p) convention.
    published = np.zeros_like(observed)
    for p in range(nm):
        for j in range(nl):
            row = p * nl + j
            if p + 1 < nm:
                published[row, (p + 1) * nl + j] -= 1j * kpar * np.sqrt(tau * (p + 1))
            if p > 0:
                published[row, (p - 1) * nl + j] -= 1j * kpar * np.sqrt(tau * p)
            perpendicular_drive = (
                2 * j * kernels[j]
                - (j * kernels[j - 1] if j else 0.0)
                - (j + 1) * kernels[j + 1]
            )
            field_drive = 1j * kperp * (
                (kernels[j] if p == 0 else 0.0)
                + eta
                * (
                    (kernels[j] / np.sqrt(2.0) if p == 2 else 0.0)
                    + (perpendicular_drive if p == 0 else 0.0)
                )
            ) - (1j * kpar * np.sqrt(tau) * kernels[j] if p == 1 else 0.0)
            for source_j in range(nl):
                published[row, source_j] += (
                    field_drive * kernels[source_j] / denominator
                )

    convention = np.zeros_like(observed.real)
    for p in range(nm):
        for j in range(nl):
            convention[j * nm + p, p * nl + j] = (-1.0) ** j
    expected = convention @ published @ convention.T
    np.testing.assert_allclose(observed, expected, rtol=5.0e-6, atol=2.0e-6)


def test_quasineutrality_simple():
    """Quasineutrality should reduce to a simple ratio for a single mode."""
    Nl, Nm, Ny, Nx, Nz = 2, 2, 1, 1, 1
    b = jnp.array([[[0.5]]])
    Jl_single = J_l_all(b, l_max=Nl - 1)
    Jl = Jl_single[None, ...]
    G = jnp.zeros((1, Nl, Nm, Ny, Nx, Nz))
    G = G.at[0, 0, 0, 0, 0, 0].set(2.0)
    phi = quasineutrality_phi(
        G,
        Jl,
        tau_e=1.0,
        charge=jnp.array([1.0]),
        density=jnp.array([1.0]),
        tz=jnp.array([1.0]),
    )
    den = 1.0 + 1.0 - jnp.sum(Jl_single[:, 0, 0, 0] ** 2)
    assert jnp.isclose(phi[0, 0, 0], Jl_single[0, 0, 0, 0] * 2.0 / den)


def test_quasineutrality_charge_sign():
    """Charge sign should flip the quasineutrality solution."""
    Nl, Nm, Ny, Nx, Nz = 2, 1, 1, 1, 4
    Jl = jnp.ones((1, Nl, Ny, Nx, Nz))
    G = jnp.zeros((1, Nl, Nm, Ny, Nx, Nz))
    G = G.at[0, 0, 0, 0, 0, :].set(1.0)
    phi_pos = quasineutrality_phi(
        G,
        Jl,
        tau_e=1.0,
        charge=jnp.array([1.0]),
        density=jnp.array([1.0]),
        tz=jnp.array([1.0]),
    )
    phi_neg = quasineutrality_phi(
        G,
        Jl,
        tau_e=1.0,
        charge=jnp.array([-1.0]),
        density=jnp.array([1.0]),
        tz=jnp.array([-1.0]),
    )
    assert jnp.allclose(phi_pos, -phi_neg)


def test_build_H_adds_phi_to_m0():
    """H should add J_l phi only to the m=0 Hermite index."""
    G = jnp.zeros((1, 2, 2, 1, 1, 1))
    Jl = jnp.ones((1, 2, 1, 1, 1))
    phi = jnp.array([[[3.0]]])
    H = build_H(G, Jl, phi, tz=jnp.array([1.0]))
    assert jnp.allclose(H[0, :, 0, 0, 0, 0], 3.0)
    assert jnp.allclose(H[0, :, 1, 0, 0, 0], 0.0)


def test_build_H_adds_apar_to_m1():
    """Apar enters H at m=1 with GX sign convention."""
    G = jnp.zeros((1, 2, 2, 1, 1, 1))
    Jl = jnp.ones((1, 2, 1, 1, 1))
    phi = jnp.zeros((1, 1, 1))
    apar = jnp.ones((1, 1, 1))
    H = build_H(G, Jl, phi, tz=jnp.array([1.0]), apar=apar, vth=jnp.array([2.0]))
    assert jnp.allclose(H[0, :, 1, 0, 0, 0], -2.0)


def test_collisions_include_low_order_conservation_correction():
    G = jnp.zeros((1, 1, 3, 1, 1, 1), dtype=jnp.complex64)
    G = G.at[0, 0, 0, 0, 0, 0].set(2.0 + 0.0j)
    G = G.at[0, 0, 1, 0, 0, 0].set(3.0 + 0.0j)
    G = G.at[0, 0, 2, 0, 0, 0].set(5.0 + 0.0j)
    Jl = jnp.ones((1, 1, 1, 1, 1), dtype=jnp.float32)
    JlB = jnp.ones((1, 1, 1, 1, 1), dtype=jnp.float32)
    H = G
    out = collisions_contribution(
        H,
        G=G,
        Jl=Jl,
        JlB=JlB,
        b=jnp.full((1, 1, 1, 1), 4.0, dtype=jnp.float32),
        nu=jnp.array([0.5], dtype=jnp.float32),
        collision_lam=jnp.zeros((1, 1, 3, 1, 1, 1), dtype=jnp.float32),
        weight=jnp.asarray(1.0, dtype=jnp.float32),
    )

    assert jnp.allclose(out[0, 0, 0, 0, 0, 0], 4.0)
    assert jnp.allclose(out[0, 0, 1, 0, 0, 0], 1.5)
    assert jnp.allclose(out[0, 0, 2, 0, 0, 0], 5.0)


def test_long_wavelength_collision_invariants_and_free_energy_rate():
    """The Mandell-Dorland-Landreman model conserves fluid moments at b=0."""

    shape = (2, 3, 5, 1, 1, 2)
    state = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)
    state = state.astype(jnp.complex64) + 0.1j
    Jl = jnp.zeros((2, 3, 1, 1, 2), dtype=jnp.float32).at[:, 0].set(1.0)
    JlB = Jl.at[:, 1].set(1.0)
    eigenvalues = jnp.asarray(
        [[2 * ell + m for m in range(5)] for ell in range(3)], dtype=jnp.float32
    )
    contribution = collisions_contribution(
        state,
        G=state,
        Jl=Jl,
        JlB=JlB,
        b=jnp.zeros((2, 1, 1, 2), dtype=jnp.float32),
        nu=jnp.asarray([0.2, 0.35], dtype=jnp.float32),
        lb_lam=eigenvalues,
        weight=jnp.asarray(1.0, dtype=jnp.float32),
    )

    rates = collision_invariant_rates(contribution)
    np.testing.assert_allclose(np.asarray(rates.density), 0.0, atol=2.0e-5)
    np.testing.assert_allclose(np.asarray(rates.parallel_momentum), 0.0, atol=2.0e-5)
    np.testing.assert_allclose(np.asarray(rates.thermal_energy), 0.0, atol=2.0e-5)
    quadratic_rate = collision_quadratic_rate(state, contribution)
    assert float(quadratic_rate) < 0.0

    def apply_collision(value):
        return collisions_contribution(
            value,
            G=value,
            Jl=Jl,
            JlB=JlB,
            b=jnp.zeros((2, 1, 1, 2), dtype=jnp.float32),
            nu=jnp.asarray([0.2, 0.35], dtype=jnp.float32),
            lb_lam=eigenvalues,
            weight=jnp.asarray(1.0, dtype=jnp.float32),
        )

    probe = jnp.flip(state, axis=(1, 2)) + 0.2j
    np.testing.assert_allclose(
        np.asarray(jnp.vdot(state, apply_collision(probe))),
        np.asarray(jnp.vdot(contribution, probe)),
        rtol=2.0e-6,
        atol=2.0e-5,
    )
    rate_gradient = jax.grad(
        lambda scale: collision_quadratic_rate(
            scale * state, apply_collision(scale * state)
        )
    )(jnp.asarray(1.0, dtype=jnp.float32))
    np.testing.assert_allclose(
        np.asarray(rate_gradient), 2.0 * np.asarray(quadratic_rate), rtol=2.0e-6
    )


def test_collision_geometry_tangent_is_finite_at_zero_wavelength():
    """The finite-Larmor correction is smooth in b at the spectral zero mode."""

    shape = (1, 3, 5, 1, 1, 1)
    state = (
        jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape) + 0.2j
    ).astype(jnp.complex64)
    eigenvalues = jnp.asarray(
        [[2 * ell + m for m in range(5)] for ell in range(3)], dtype=jnp.float32
    )

    def loss(b_value):
        b = jnp.full((1, 1, 1, 1), b_value)
        Jl = jnp.moveaxis(J_l_all(b, 2), 0, 1)
        Jl_m1 = jnp.pad(Jl[:, :-1], ((0, 0), (1, 0), (0, 0), (0, 0), (0, 0)))
        contribution = collisions_contribution(
            state,
            G=state,
            Jl=Jl,
            JlB=Jl + Jl_m1,
            b=b,
            nu=jnp.asarray([0.01], dtype=jnp.float32),
            lb_lam=eigenvalues,
            weight=jnp.asarray(1.0, dtype=jnp.float32),
        )
        return jnp.real(jnp.vdot(contribution, contribution))

    tangent = jax.grad(loss)(jnp.asarray(0.0, dtype=jnp.float32))
    step = jnp.asarray(1.0e-4, dtype=jnp.float32)
    forward_difference = (loss(step) - loss(0.0)) / step
    assert jnp.isfinite(tangent)
    np.testing.assert_allclose(tangent, forward_difference, rtol=2.0e-3, atol=2.0e-3)


def test_long_wavelength_collision_matches_published_dougherty_equation_c6():
    """Production moments match Frei et al. (2022), Appendix C, equation C6."""

    shape = (2, 3, 5, 1, 1, 2)
    state = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)
    state = state.astype(jnp.complex64) + 0.17j
    nu = jnp.asarray([0.2, 0.35], dtype=jnp.float32)
    reference = drift_kinetic_dougherty_contribution(state, nu=nu)

    Jl = jnp.zeros((2, 3, 1, 1, 2), dtype=jnp.float32).at[:, 0].set(1.0)
    production = collisions_contribution(
        state,
        G=state,
        Jl=Jl,
        JlB=Jl.at[:, 1].set(1.0),
        b=jnp.zeros((2, 1, 1, 2), dtype=jnp.float32),
        nu=nu,
        lb_lam=jnp.asarray(
            [[2 * ell + m for m in range(5)] for ell in range(3)],
            dtype=jnp.float32,
        ),
        weight=jnp.asarray(1.0, dtype=jnp.float32),
    )
    np.testing.assert_allclose(
        np.asarray(production), np.asarray(reference), rtol=2.0e-6, atol=2.0e-5
    )

    rates = collision_invariant_rates(reference)
    np.testing.assert_allclose(np.asarray(rates.density), 0.0, atol=2.0e-6)
    np.testing.assert_allclose(np.asarray(rates.parallel_momentum), 0.0, atol=2.0e-6)
    np.testing.assert_allclose(np.asarray(rates.thermal_energy), 0.0, atol=2.0e-5)
    assert float(collision_quadratic_rate(state, reference)) < 0.0

    tangent = jax.jvp(
        lambda frequency: drift_kinetic_dougherty_contribution(state, nu=frequency),
        (nu,),
        (jnp.ones_like(nu),),
    )[1]
    step = jnp.asarray(1.0e-3, dtype=jnp.float32)
    finite_difference = (
        drift_kinetic_dougherty_contribution(state, nu=nu + step)
        - drift_kinetic_dougherty_contribution(state, nu=nu - step)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        np.asarray(tangent),
        np.asarray(finite_difference),
        rtol=2.0e-4,
        atol=2.0e-3,
    )
    np.testing.assert_allclose(
        np.asarray(drift_kinetic_dougherty_contribution(state[0], nu=nu[:1])),
        np.asarray(reference[0]),
        rtol=2.0e-6,
        atol=2.0e-5,
    )
    with pytest.raises(ValueError, match="Nl >= 2"):
        drift_kinetic_dougherty_contribution(state[:, :1], nu=nu)
    with pytest.raises(ValueError, match="five or six"):
        drift_kinetic_dougherty_contribution(jnp.ones((2, 3)), nu=nu)


def test_long_wavelength_local_maxwellian_is_collision_null_space():
    state = jnp.zeros((1, 2, 3, 1, 1, 1), dtype=jnp.complex64)
    state = state.at[:, 0, 0].set(1.2)
    state = state.at[:, 0, 1].set(-0.4)
    state = state.at[:, 0, 2].set(0.7 / jnp.sqrt(2.0))
    state = state.at[:, 1, 0].set(0.7)
    Jl = jnp.zeros((1, 2, 1, 1, 1), dtype=jnp.float32).at[:, 0].set(1.0)
    JlB = Jl.at[:, 1].set(1.0)
    contribution = collisions_contribution(
        state,
        G=state,
        Jl=Jl,
        JlB=JlB,
        b=jnp.zeros((1, 1, 1, 1), dtype=jnp.float32),
        nu=jnp.asarray([0.3], dtype=jnp.float32),
        lb_lam=jnp.asarray([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]]),
        weight=jnp.asarray(1.0, dtype=jnp.float32),
    )

    np.testing.assert_allclose(np.asarray(contribution), 0.0, atol=2.0e-7)


def test_sugama_six_moment_matrix_matches_published_equation_and_invariants():
    """Frei, Ernst & Ricci (2022), Appendix C, equations (C6a)--(C6f)."""

    state = jnp.zeros((2, 2, 4, 1, 1, 1), dtype=jnp.complex64)
    state = state.at[:, 0, 0].set(jnp.asarray([1.2, -0.4])[:, None, None, None])
    state = state.at[:, 0, 1].set(jnp.asarray([0.3, 0.7])[:, None, None, None])
    state = state.at[:, 0, 2].set(jnp.asarray([0.8, -0.2])[:, None, None, None])
    state = state.at[:, 1, 0].set(jnp.asarray([-0.5, 0.9])[:, None, None, None])
    state = state.at[:, 0, 3].set(jnp.asarray([0.6, -0.4])[:, None, None, None])
    state = state.at[:, 1, 1].set(jnp.asarray([0.2, 0.7])[:, None, None, None])
    nu = jnp.asarray([0.2, 0.35], dtype=jnp.float32)
    result = drift_kinetic_sugama_six_moment_contribution(state, nu=nu)

    sqrt_two_over_pi = np.sqrt(2.0 / np.pi)
    sqrt_one_over_pi = np.sqrt(1.0 / np.pi)
    sqrt_one_over_three_pi = np.sqrt(1.0 / (3.0 * np.pi))
    matrix = np.asarray(
        [
            [
                -(64.0 / 45.0) * sqrt_two_over_pi,
                (64.0 / 45.0) * sqrt_one_over_pi,
                0.0,
                0.0,
            ],
            [
                (64.0 / 45.0) * sqrt_one_over_pi,
                -(32.0 / 45.0) * sqrt_two_over_pi,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                -(361.0 / 175.0) * sqrt_two_over_pi,
                (208.0 / 175.0) * sqrt_one_over_three_pi,
            ],
            [
                0.0,
                0.0,
                (208.0 / 175.0) * sqrt_one_over_three_pi,
                -(1187.0 / 525.0) * sqrt_two_over_pi,
            ],
        ],
        dtype=np.float32,
    )
    for species in range(2):
        moments = np.asarray(
            [
                state[species, 0, 2, 0, 0, 0],
                state[species, 1, 0, 0, 0, 0],
                state[species, 0, 3, 0, 0, 0],
                state[species, 1, 1, 0, 0, 0],
            ]
        )
        expected = float(nu[species]) * matrix @ moments
        actual = np.asarray(
            [
                result[species, 0, 2, 0, 0, 0],
                result[species, 1, 0, 0, 0, 0],
                result[species, 0, 3, 0, 0, 0],
                result[species, 1, 1, 0, 0, 0],
            ]
        )
        np.testing.assert_allclose(actual, expected, rtol=2.0e-6, atol=2.0e-7)

    rates = collision_invariant_rates(result)
    np.testing.assert_allclose(np.asarray(rates.density), 0.0, atol=2.0e-7)
    np.testing.assert_allclose(np.asarray(rates.parallel_momentum), 0.0, atol=2.0e-7)
    np.testing.assert_allclose(np.asarray(rates.thermal_energy), 0.0, atol=2.0e-7)
    np.testing.assert_allclose(matrix, matrix.T, atol=1.0e-7)
    assert np.linalg.eigvalsh(matrix).max() < 2.0e-7
    assert float(collision_quadratic_rate(state, result)) < 0.0


def test_sugama_six_moment_null_space_and_collision_frequency_derivative():
    state = jnp.zeros((1, 2, 4, 1, 1, 1), dtype=jnp.complex64)
    state = state.at[:, 0, 0].set(1.0)
    state = state.at[:, 0, 1].set(-0.3)
    state = state.at[:, 0, 2].set(1.0)
    state = state.at[:, 1, 0].set(jnp.sqrt(2.0))
    nu = jnp.asarray([0.3], dtype=jnp.float32)
    null_result = drift_kinetic_sugama_six_moment_contribution(state, nu=nu)
    np.testing.assert_allclose(np.asarray(null_result), 0.0, atol=2.0e-7)

    probe = state.at[:, 0, 3].set(0.6).at[:, 1, 1].set(-0.2)
    tangent = jax.jvp(
        lambda frequency: drift_kinetic_sugama_six_moment_contribution(
            probe, nu=frequency
        ),
        (nu,),
        (jnp.ones_like(nu),),
    )[1]
    step = jnp.asarray(1.0e-3, dtype=jnp.float32)
    finite_difference = (
        drift_kinetic_sugama_six_moment_contribution(probe, nu=nu + step)
        - drift_kinetic_sugama_six_moment_contribution(probe, nu=nu - step)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        np.asarray(tangent), np.asarray(finite_difference), rtol=2.0e-4, atol=2.0e-5
    )
    np.testing.assert_allclose(
        np.asarray(drift_kinetic_sugama_six_moment_contribution(probe[0], nu=nu)),
        np.asarray(drift_kinetic_sugama_six_moment_contribution(probe, nu=nu)[0]),
        rtol=2.0e-6,
        atol=2.0e-7,
    )
    with pytest.raises(ValueError, match="Nl >= 2 and Nm >= 4"):
        drift_kinetic_sugama_six_moment_contribution(probe[:, :, :3], nu=nu)
    with pytest.raises(ValueError, match="five or six"):
        drift_kinetic_sugama_six_moment_contribution(jnp.ones((2, 4)), nu=nu)


def test_coulomb_six_moment_matrix_matches_published_equation_c9():
    state = jnp.zeros((1, 2, 4, 1, 1, 1), dtype=jnp.complex64)
    state = state.at[:, 0, 2].set(0.8)
    state = state.at[:, 1, 0].set(-0.5)
    state = state.at[:, 0, 3].set(0.6)
    state = state.at[:, 1, 1].set(0.2)
    result = drift_kinetic_coulomb_six_moment_contribution(state, nu=jnp.asarray([0.2]))
    inverse_sqrt_pi = 1.0 / np.sqrt(np.pi)
    matrix = inverse_sqrt_pi * np.asarray(
        [
            [-16.0 * np.sqrt(2.0) / 15.0, 16.0 / 15.0, 0.0, 0.0],
            [16.0 / 15.0, -8.0 * np.sqrt(2.0) / 15.0, 0.0, 0.0],
            [0.0, 0.0, -8.0 * np.sqrt(2.0) / 5.0, 8.0 / (5.0 * np.sqrt(3.0))],
            [0.0, 0.0, 8.0 / (5.0 * np.sqrt(3.0)), -28.0 * np.sqrt(2.0) / 15.0],
        ],
        dtype=np.float32,
    )
    moments = np.asarray(
        [
            state[0, 0, 2, 0, 0, 0],
            state[0, 1, 0, 0, 0, 0],
            state[0, 0, 3, 0, 0, 0],
            state[0, 1, 1, 0, 0, 0],
        ]
    )
    actual = np.asarray(
        [
            result[0, 0, 2, 0, 0, 0],
            result[0, 1, 0, 0, 0, 0],
            result[0, 0, 3, 0, 0, 0],
            result[0, 1, 1, 0, 0, 0],
        ]
    )
    np.testing.assert_allclose(actual, 0.2 * matrix @ moments, rtol=2.0e-6)
    np.testing.assert_allclose(matrix, matrix.T, atol=1.0e-7)
    assert np.linalg.eigvalsh(matrix).max() < 2.0e-7
    rates = collision_invariant_rates(result)
    np.testing.assert_allclose(np.asarray(rates.thermal_energy), 0.0, atol=2.0e-7)
    assert float(collision_quadratic_rate(state, result)) < 0.0


def test_finite_larmor_collision_matches_published_moment_equations():
    """Check Mandell et al. (2018), equations (3.38)--(3.42) and (4.10)."""

    nl, nm = 4, 5
    b = jnp.asarray([[[[0.7]]]], dtype=jnp.float32)
    Jl = jnp.moveaxis(J_l_all(b, nl - 1), 0, 1)
    Jl_m1 = jnp.pad(Jl[:, :-1], ((0, 0), (1, 0), (0, 0), (0, 0), (0, 0)))
    Jl_p1 = jnp.pad(Jl[:, 1:], ((0, 0), (0, 1), (0, 0), (0, 0), (0, 0)))
    JlB = Jl + Jl_m1
    ell = jnp.arange(nl, dtype=jnp.float32)[None, :, None, None, None]
    temperature_coeff = ell * Jl_m1 + 2.0 * ell * Jl + (ell + 1.0) * Jl_p1
    state = (
        jnp.arange(nl * nm, dtype=jnp.float32).reshape(1, nl, nm, 1, 1, 1) + 0.2j
    ).astype(jnp.complex64)
    nu = jnp.asarray([0.3], dtype=jnp.float32)

    u_parallel = jnp.sum(Jl * state[:, :, 1], axis=1)
    u_perpendicular = jnp.sqrt(b) * jnp.sum(JlB * state[:, :, 0], axis=1)
    temperature = jnp.sqrt(2.0) / 3.0 * jnp.sum(
        Jl * state[:, :, 2], axis=1
    ) + 2.0 / 3.0 * jnp.sum(temperature_coeff * state[:, :, 0], axis=1)
    eigenvalues = jnp.asarray(
        [[0.7 + 2 * ell_index + m for m in range(nm)] for ell_index in range(nl)],
        dtype=jnp.float32,
    )
    expected = (
        -nu[:, None, None, None, None, None]
        * eigenvalues[None, :, :, None, None, None]
        * state
    )
    expected = expected.at[:, :, 0].add(
        nu[:, None, None, None, None]
        * (
            jnp.sqrt(b) * JlB * u_perpendicular[:, None]
            + 2.0 * temperature_coeff * temperature[:, None]
        )
    )
    expected = expected.at[:, :, 1].add(
        nu[:, None, None, None, None] * Jl * u_parallel[:, None]
    )
    expected = expected.at[:, :, 2].add(
        nu[:, None, None, None, None] * jnp.sqrt(2.0) * Jl * temperature[:, None]
    )
    contribution = collisions_contribution(
        state,
        G=state,
        Jl=Jl,
        JlB=JlB,
        b=b,
        nu=nu,
        lb_lam=jnp.asarray(
            [[2 * ell_index + m for m in range(nm)] for ell_index in range(nl)],
            dtype=jnp.float32,
        ),
        weight=jnp.asarray(1.0, dtype=jnp.float32),
    )

    np.testing.assert_allclose(
        np.asarray(contribution), np.asarray(expected), rtol=2.0e-6, atol=2.0e-6
    )
    quadratic_rate = collision_quadratic_rate(state, contribution)
    differential_rate = jnp.sum(
        nu[:, None, None, None, None, None]
        * eigenvalues[None, :, :, None, None, None]
        * jnp.abs(state) ** 2
    )
    restoring_rate = jnp.sum(
        nu[:, None, None, None, None]
        * (
            jnp.abs(u_parallel) ** 2
            + jnp.abs(u_perpendicular) ** 2
            + 3.0 * jnp.abs(temperature) ** 2
        )
    )
    np.testing.assert_allclose(
        np.asarray(quadratic_rate),
        np.asarray(-(differential_rate - restoring_rate)),
        rtol=2.0e-6,
        atol=1.0e-3,
    )
    assert float(quadratic_rate) < 0.0


def test_finite_larmor_collision_has_first_order_drift_kinetic_limit():
    """The Mandell finite-b model approaches Frei et al. equation (C6)."""

    nl, nm = 4, 5
    state = (
        jnp.arange(nl * nm, dtype=jnp.float32).reshape(1, nl, nm, 1, 1, 1) + 0.2j
    ).astype(jnp.complex64)
    nu = jnp.asarray([0.3], dtype=jnp.float32)
    eigenvalues = jnp.asarray(
        [[2 * ell + m for m in range(nm)] for ell in range(nl)],
        dtype=jnp.float32,
    )
    drift_kinetic = drift_kinetic_dougherty_contribution(state, nu=nu)

    errors = []
    b_values = (0.2, 0.1, 0.05, 0.025)
    for b_value in b_values:
        b = jnp.asarray([[[[b_value]]]], dtype=jnp.float32)
        Jl = jnp.moveaxis(J_l_all(b, nl - 1), 0, 1)
        Jl_m1 = jnp.pad(Jl[:, :-1], ((0, 0), (1, 0), (0, 0), (0, 0), (0, 0)))
        finite_larmor = collisions_contribution(
            state,
            G=state,
            Jl=Jl,
            JlB=Jl + Jl_m1,
            b=b,
            nu=nu,
            lb_lam=eigenvalues,
            weight=jnp.asarray(1.0, dtype=jnp.float32),
        )
        errors.append(
            float(
                jnp.linalg.norm(finite_larmor - drift_kinetic)
                / jnp.linalg.norm(drift_kinetic)
            )
        )

    order = estimate_observed_order(
        np.asarray(b_values, dtype=float), np.asarray(errors, dtype=float)
    ).asymptotic_order
    assert 0.85 <= order <= 1.15
    assert all(fine < coarse for coarse, fine in zip(errors, errors[1:]))


def test_collision_diagnostics_validate_shapes_and_weights():
    state = jnp.ones((2, 3, 1, 1, 1), dtype=jnp.complex64)
    contribution = -state
    rate = collision_quadratic_rate(state, contribution, weights=2.0)
    np.testing.assert_allclose(np.asarray(rate), -12.0)
    with pytest.raises(ValueError, match="same shape"):
        collision_quadratic_rate(state, contribution[..., 0])
    with pytest.raises(ValueError, match="five or six"):
        collision_invariant_rates(jnp.ones((2, 3, 1)))
    with pytest.raises(ValueError, match="Nl >= 2"):
        collision_invariant_rates(jnp.ones((1, 2, 1, 1, 1)))


def test_multispecies_collision_diagnostic_uses_physical_moment_weights():
    """Cross-species gates conserve particles and total momentum/energy."""

    contribution = jnp.zeros((2, 2, 3, 1, 1, 1), dtype=jnp.complex64)
    density = jnp.asarray([2.0, 0.5], dtype=jnp.float32)
    mass = jnp.asarray([4.0, 1.0], dtype=jnp.float32)
    temperature = jnp.asarray([1.0, 9.0], dtype=jnp.float32)
    momentum_weights = density * jnp.sqrt(mass * temperature)
    energy_weights = density * temperature

    contribution = contribution.at[0, 0, 1].set(0.75)
    contribution = contribution.at[1, 0, 1].set(
        -0.75 * momentum_weights[0] / momentum_weights[1]
    )
    contribution = contribution.at[0, 0, 2].set(0.4 / jnp.sqrt(2.0))
    contribution = contribution.at[1, 1, 0].set(
        -0.2 * energy_weights[0] / energy_weights[1]
    )
    rates = multispecies_collision_invariant_rates(
        contribution, density=density, mass=mass, temperature=temperature
    )

    np.testing.assert_allclose(np.asarray(rates.particle_density), 0.0, atol=1.0e-7)
    np.testing.assert_allclose(
        np.asarray(rates.total_parallel_momentum), 0.0, atol=1.0e-7
    )
    np.testing.assert_allclose(np.asarray(rates.total_thermal_energy), 0.0, atol=1.0e-7)
    energy_gradient = jax.grad(
        lambda scale: jnp.real(
            multispecies_collision_invariant_rates(
                scale * contribution,
                density=density,
                mass=mass,
                temperature=temperature,
            ).total_thermal_energy.sum()
        )
    )(jnp.asarray(1.0, dtype=jnp.float32))
    np.testing.assert_allclose(np.asarray(energy_gradient), 0.0, atol=1.0e-7)

    with pytest.raises(ValueError, match="mass must have length 2"):
        multispecies_collision_invariant_rates(
            contribution,
            density=density,
            mass=jnp.asarray([1.0]),
            temperature=temperature,
        )


def test_full_f_dougherty_cross_moments_satisfy_pairwise_conservation_and_ad() -> None:
    """Francisquez et al. (2022), equations (2.11)--(2.12)."""

    density = jnp.asarray([1.0, 0.8], dtype=jnp.float32)
    mass = jnp.asarray([4.0, 1.0], dtype=jnp.float32)
    flow = jnp.asarray([0.3, -0.2], dtype=jnp.float32)
    thermal = jnp.asarray([0.7, 1.4], dtype=jnp.float32)
    nu = jnp.asarray([[0.0, 0.25], [0.6, 0.0]], dtype=jnp.float32)
    targets = conservative_full_f_dougherty_cross_moments(
        flow,
        thermal,
        density=density,
        mass=mass,
        collision_frequency=nu,
    )

    momentum_rate_sr = mass[0] * density[0] * nu[0, 1]
    momentum_rate_rs = mass[1] * density[1] * nu[1, 0]
    expected_flow = (momentum_rate_sr * flow[0] + momentum_rate_rs * flow[1]) / (
        momentum_rate_sr + momentum_rate_rs
    )
    expected_thermal = (
        mass[0] * density[0] * nu[0, 1] * thermal[0]
        + mass[1] * density[1] * nu[1, 0] * thermal[1]
        + momentum_rate_sr
        * momentum_rate_rs
        / (momentum_rate_sr + momentum_rate_rs)
        * (flow[0] - flow[1]) ** 2
        / 3.0
    ) / (mass[0] * (density[0] * nu[0, 1] + density[1] * nu[1, 0]))
    np.testing.assert_allclose(targets.parallel_flow[0, 1], expected_flow)
    np.testing.assert_allclose(targets.parallel_flow[1, 0], expected_flow)
    np.testing.assert_allclose(targets.thermal_speed_sq[0, 1], expected_thermal)
    np.testing.assert_allclose(
        mass[1] * targets.thermal_speed_sq[1, 0],
        mass[0] * expected_thermal,
    )

    momentum_rate = mass * density * nu.sum(axis=1)
    momentum_change = momentum_rate * (
        jnp.asarray([targets.parallel_flow[0, 1], targets.parallel_flow[1, 0]]) - flow
    )
    np.testing.assert_allclose(momentum_change.sum(), 0.0, atol=2.0e-7)

    target_flow = jnp.asarray(
        [targets.parallel_flow[0, 1], targets.parallel_flow[1, 0]]
    )
    target_thermal = jnp.asarray(
        [targets.thermal_speed_sq[0, 1], targets.thermal_speed_sq[1, 0]]
    )
    energy_change = (
        mass
        * density
        * nu.sum(axis=1)
        * (3.0 * (target_thermal - thermal) + flow * (target_flow - flow))
    )
    np.testing.assert_allclose(energy_change.sum(), 0.0, atol=3.0e-7)
    np.testing.assert_allclose(target_flow[0], target_flow[1], atol=1.0e-7)
    np.testing.assert_allclose(
        mass[0] * target_thermal[0], mass[1] * target_thermal[1], atol=2.0e-7
    )
    assert bool(jnp.all(target_thermal > 0.0))

    derivative = jax.grad(
        lambda u0: conservative_full_f_dougherty_cross_moments(
            flow.at[0].set(u0),
            thermal,
            density=density,
            mass=mass,
            collision_frequency=nu,
        ).thermal_speed_sq[0, 1]
    )(flow[0])
    centered = (
        conservative_full_f_dougherty_cross_moments(
            flow.at[0].add(1.0e-3),
            thermal,
            density=density,
            mass=mass,
            collision_frequency=nu,
        ).thermal_speed_sq[0, 1]
        - conservative_full_f_dougherty_cross_moments(
            flow.at[0].add(-1.0e-3),
            thermal,
            density=density,
            mass=mass,
            collision_frequency=nu,
        ).thermal_speed_sq[0, 1]
    ) / 2.0e-3
    np.testing.assert_allclose(derivative, centered, rtol=2.0e-4, atol=2.0e-5)


def test_full_f_dougherty_cross_moments_equal_species_limit_and_shapes() -> None:
    flow = jnp.asarray([0.4, -0.2])
    thermal = jnp.asarray([0.6, 1.0])
    targets = conservative_full_f_dougherty_cross_moments(
        flow,
        thermal,
        density=jnp.ones(2),
        mass=jnp.ones(2),
        collision_frequency=jnp.asarray([[0.0, 0.3], [0.3, 0.0]]),
    )
    expected_flow = 0.5 * (flow[0] + flow[1])
    expected_thermal = 0.5 * (thermal[0] + thermal[1] + (flow[0] - flow[1]) ** 2 / 6.0)
    np.testing.assert_allclose(targets.parallel_flow[0, 1], expected_flow)
    np.testing.assert_allclose(targets.thermal_speed_sq[0, 1], expected_thermal)
    np.testing.assert_allclose(targets.parallel_flow.diagonal(), flow)
    np.testing.assert_allclose(targets.thermal_speed_sq.diagonal(), thermal)

    with pytest.raises(ValueError, match="collision_frequency must have shape"):
        conservative_full_f_dougherty_cross_moments(
            flow,
            thermal,
            density=jnp.ones(2),
            mass=jnp.ones(2),
            collision_frequency=jnp.ones((2, 1)),
        )
    with pytest.raises(ValueError, match="collision_frequency must be non-negative"):
        conservative_full_f_dougherty_cross_moments(
            flow,
            thermal,
            density=jnp.ones(2),
            mass=jnp.ones(2),
            collision_frequency=jnp.asarray([[0.0, -0.1], [0.1, 0.0]]),
        )


@pytest.mark.parametrize("velocity_dimensions", [1, 2, 3])
def test_full_f_dougherty_cross_moments_conserve_each_multispecies_pair(
    velocity_dimensions: int,
) -> None:
    """Francisquez equations (2.11)--(2.12) conserve every species pair."""

    flow = jnp.asarray(
        [[0.3, -0.2, 0.5], [-0.4, 0.1, 0.2], [0.8, -0.5, 0.0]],
        dtype=jnp.float32,
    )
    thermal = jnp.asarray(
        [[0.7, 0.9, 1.1], [1.2, 0.6, 0.5], [0.4, 1.5, 0.8]],
        dtype=jnp.float32,
    )
    density = jnp.asarray([1.0, 0.8, 1.3], dtype=jnp.float32)
    mass = jnp.asarray([4.0, 1.0, 2.0], dtype=jnp.float32)
    frequency = jnp.asarray(
        [[0.0, 0.20, 0.10], [0.35, 0.0, 0.15], [0.22, 0.18, 0.0]],
        dtype=jnp.float32,
    )
    targets = conservative_full_f_dougherty_cross_moments(
        flow,
        thermal,
        density=density,
        mass=mass,
        collision_frequency=frequency,
        velocity_dimensions=velocity_dimensions,
    )

    for species in range(3):
        for partner in range(species + 1, 3):
            rate_s = mass[species] * density[species] * frequency[species, partner]
            rate_r = mass[partner] * density[partner] * frequency[partner, species]
            momentum_change = rate_s * (
                targets.parallel_flow[species, partner] - flow[species]
            ) + rate_r * (targets.parallel_flow[partner, species] - flow[partner])
            energy_change = rate_s * (
                velocity_dimensions
                * (targets.thermal_speed_sq[species, partner] - thermal[species])
                + flow[species]
                * (targets.parallel_flow[species, partner] - flow[species])
            ) + rate_r * (
                velocity_dimensions
                * (targets.thermal_speed_sq[partner, species] - thermal[partner])
                + flow[partner]
                * (targets.parallel_flow[partner, species] - flow[partner])
            )
            np.testing.assert_allclose(momentum_change, 0.0, atol=2.0e-7)
            np.testing.assert_allclose(energy_change, 0.0, atol=3.0e-7)

    offset = jnp.asarray(1.7, dtype=flow.dtype)
    shifted = conservative_full_f_dougherty_cross_moments(
        flow + offset,
        thermal,
        density=density,
        mass=mass,
        collision_frequency=frequency,
        velocity_dimensions=velocity_dimensions,
    )
    np.testing.assert_allclose(
        shifted.parallel_flow, targets.parallel_flow + offset, atol=2.0e-7
    )
    np.testing.assert_allclose(
        shifted.thermal_speed_sq, targets.thermal_speed_sq, atol=2.0e-7
    )


def test_collisions_contribution_accepts_low_rank_lb_lam():
    H = jnp.ones((1, 1, 2, 1, 1, 1), dtype=jnp.complex64)
    out = collisions_contribution(
        H,
        nu=jnp.array([0.5], dtype=jnp.float32),
        lb_lam=jnp.array([[0.0, 1.0]], dtype=jnp.float32),
        b=jnp.full((1, 1, 1, 1), 2.0, dtype=jnp.float32),
        weight=jnp.asarray(1.0, dtype=jnp.float32),
    )

    np.testing.assert_allclose(np.asarray(out[0, 0, :, 0, 0, 0]), [-1.0, -1.5])


def test_collisions_contribution_skips_zero_nu_low_rank_path():
    H = jnp.ones((1, 1, 2, 1, 1, 1), dtype=jnp.complex64)
    out = collisions_contribution(
        H,
        nu=jnp.array([0.0], dtype=jnp.float32),
        lb_lam=jnp.array([[2.0, 3.0]], dtype=jnp.float32),
        b=jnp.full((1, 1, 1, 1), 2.0, dtype=jnp.float32),
        weight=jnp.asarray(1.0, dtype=jnp.float32),
    )

    assert jnp.allclose(out, jnp.zeros_like(H))


def test_collisions_contribution_preserves_preexpanded_collision_lam_when_nu_zero():
    H = jnp.ones((1, 1, 2, 1, 1, 1), dtype=jnp.complex64)
    out = collisions_contribution(
        H,
        nu=jnp.array([0.0], dtype=jnp.float32),
        collision_lam=jnp.full((1, 1, 2, 1, 1, 1), 2.0, dtype=jnp.float32),
        weight=jnp.asarray(1.0, dtype=jnp.float32),
    )

    assert jnp.allclose(out, -2.0 * H)


def test_build_H_adds_bpar_to_m0():
    """Bpar term should enter H at m=0 with J_l + J_{l-1}."""
    G = jnp.zeros((1, 3, 2, 1, 1, 1))
    Jl = jnp.ones((1, 3, 1, 1, 1))
    JlB = Jl + jnp.pad(Jl[:, :-1, ...], ((0, 0), (1, 0), (0, 0), (0, 0), (0, 0)))
    phi = jnp.zeros((1, 1, 1))
    bpar = jnp.ones((1, 1, 1))
    H = build_H(G, Jl, phi, tz=jnp.array([1.0]), bpar=bpar, JlB=JlB)
    assert jnp.allclose(H[0, :, 0, 0, 0, 0], JlB[0, :, 0, 0, 0])


def test_linear_cache_bessel_bmag_power_scales_b():
    grid = build_spectral_grid(GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.0, Ly=6.0))
    common = dict(R0=2.77778, epsilon=0.18, kperp2_bmag=False)
    geometries = (
        SAlphaGeometry.from_config(GeometryConfig(**common, bessel_bmag_power=0.0)),
        SAlphaGeometry.from_config(GeometryConfig(**common, bessel_bmag_power=1.0)),
    )
    params = LinearParams()
    base, scaled = (
        build_linear_cache(grid, geom, params, Nl=2, Nm=2) for geom in geometries
    )
    bmag = geometries[0].bmag(jnp.asarray(grid.z))
    base_b, scaled_b = base.b[0, 1, 0], scaled.b[0, 1, 0]
    assert jnp.all(base_b > 0.0) and jnp.max(bmag) > jnp.min(bmag)
    assert jnp.allclose(scaled_b / base_b, 1.0 / bmag, rtol=1.0e-6, atol=1.0e-8)


def test_build_linear_cache_accepts_sampled_geometry_contract():
    grid_cfg = GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.0, Ly=6.0)
    grid = build_spectral_grid(grid_cfg)
    geom = SAlphaGeometry.from_config(GeometryConfig(R0=2.77778, epsilon=0.18))
    sampled = sample_flux_tube_geometry(geom, jnp.asarray(grid.z))
    params = LinearParams()
    cache_geom = build_linear_cache(grid, geom, params, Nl=2, Nm=2)
    cache_sampled = build_linear_cache(grid, sampled, params, Nl=2, Nm=2)
    assert jnp.allclose(cache_sampled.kperp2, cache_geom.kperp2)
    assert jnp.allclose(cache_sampled.omega_d, cache_geom.omega_d)
    assert jnp.allclose(cache_sampled.bmag, cache_geom.bmag)


def test_build_linear_cache_restores_linked_end_damping_on_full_fft_grid(spectral_grid):
    # The two-sided axis is this test's subject, not its setting: the mirror
    # row below is ``(-j) % Nky``, which names the conjugate partner only when
    # the negative rows are stored. On the half axis it names an unrelated
    # positive row, so the grid asks for ``"full"`` by name rather than
    # inheriting whichever layout ships.
    grid = spectral_grid(
        Nx=8,
        Ny=8,
        Nz=8,
        Lx=62.8,
        Ly=2.0 * np.pi,
        boundary="linked",
        y0=1.0,
        ky_layout="full",
    )
    geom = SAlphaGeometry.from_config(GeometryConfig(s_hat=0.8))
    cache = build_linear_cache(grid, geom, LinearParams(), Nl=2, Nm=2)

    profile = np.asarray(cache.linked_damp_profile, dtype=float)
    assert profile.shape == (grid.ky.size, grid.kx.size, grid.z.size)
    assert np.max(profile) > 0.0

    pos_rows = np.flatnonzero(np.max(profile, axis=(1, 2)) > 0.0)
    pos_rows = pos_rows[pos_rows > 0]
    assert pos_rows.size > 0
    ky_idx = int(pos_rows[0])
    kx_idx = int(np.flatnonzero(np.max(profile[ky_idx], axis=-1) > 0.0)[0])
    mirror_ky = (-ky_idx) % int(grid.ky.size)
    mirror_kx = 0 if kx_idx == 0 else int(grid.kx.size - kx_idx)
    assert np.allclose(profile[mirror_ky, mirror_kx], profile[ky_idx, kx_idx])


def test_build_linear_cache_keeps_linked_end_damping_on_selected_positive_ky_grid(
    spectral_grid,
):
    grid_full = spectral_grid(
        Nx=1,
        Ny=16,
        Nz=96,
        Lx=62.8,
        Ly=20.0 * np.pi,
        boundary="linked",
        y0=10.0,
        ntheta=32,
        nperiod=2,
    )
    ky_idx = int(np.argmin(np.abs(np.asarray(grid_full.ky) - 0.3)))
    assert float(grid_full.ky[ky_idx]) > 0.0
    grid = select_ky_grid(grid_full, ky_idx)
    geom = SAlphaGeometry.from_config(GeometryConfig(s_hat=0.8))
    cache = build_linear_cache(grid, geom, LinearParams(), Nl=16, Nm=48)

    profile = np.asarray(cache.linked_damp_profile, dtype=float)
    assert profile.shape == (1, grid.kx.size, grid.z.size)
    assert np.max(profile) > 0.0
    assert int(np.asarray(grid.ky_mode)[0]) > 0


@pytest.mark.parametrize("ky_layout", ["full", "half"])
@pytest.mark.parametrize("nx", [1, 4])
def test_selected_ky_hyperdiffusion_keeps_the_parent_grid_cutoff(
    only_term_config, spectral_grid, nx, ky_layout
):
    """GX normalizes ``Dfac`` by the full grid's dealiased ``k_perp^2`` corner.

    A ky scan runs every mode on a one-row :func:`select_ky_grid` slice; the
    slice has to damp each mode exactly as the full grid does, not normalize
    by its own ``ky`` and hand every mode the full ``D_hyper``.
    """

    grid_full = spectral_grid(
        Nx=nx,
        Ny=24,
        Nz=8,
        Lx=62.8,
        y0=10.0,
        boundary="periodic",
        ky_layout=ky_layout,
    )
    geom = SAlphaGeometry.from_config(GeometryConfig(s_hat=0.8))
    params = LinearParams(D_hyper=0.05, p_hyper_kperp=2.0)
    term_cfg = only_term_config(hyperdiffusion=1.0)
    G_full = jnp.ones((1, 1, int(grid_full.ky.size), nx, 8), dtype=jnp.complex128)
    cache_full = build_linear_cache(grid_full, geom, params, Nl=1, Nm=1)
    _rhs, _fields, contrib_full = assemble_rhs_terms_cached(
        G_full, cache_full, params, terms=term_cfg
    )
    hyper_full = np.asarray(contrib_full["hyperdiffusion"])
    # A linear slice unmasks every kx column (select_ky_grid); only the
    # columns the parent grid keeps exist in GX, so only those are compared.
    kept_kx = np.asarray(grid_full.dealias_mask)[0]
    for iky in (1, 3, 7):
        grid = select_ky_grid(grid_full, iky)
        cache = build_linear_cache(grid, geom, params, Nl=1, Nm=1)
        _rhs, _fields, contrib = assemble_rhs_terms_cached(
            G_full[:, :, iky : iky + 1], cache, params, terms=term_cfg
        )
        np.testing.assert_allclose(
            np.asarray(contrib["hyperdiffusion"])[..., kept_kx, :],
            hyper_full[..., iky : iky + 1, kept_kx, :],
            rtol=1.0e-12,
            atol=0.0,
        )
    # A mode well inside the cutoff is damped by (k_perp^2 / k_perp,max^2)^p,
    # a small fraction of D_hyper.
    rate = -np.real(hyper_full[0, 0, 1, 0, 0])
    assert 0.0 < rate < 0.05


def test_single_ky_linear_run_with_hyperdiffusion_matches_the_full_grid_row(
    only_terms, spectral_grid
):
    """One Euler step of the linked linear system: ky slice vs full grid row."""

    grid_full = spectral_grid(
        Nx=1,
        Ny=24,
        Nz=16,
        Lx=62.8,
        y0=10.0,
        boundary="linked",
    )
    geom = SAlphaGeometry.from_config(GeometryConfig(s_hat=0.8))
    params = LinearParams(D_hyper=0.05, p_hyper_kperp=2.0)
    terms = replace(LinearTerms(), hyperdiffusion=1.0)
    rng = np.random.default_rng(7)
    shape = (2, 3, int(grid_full.ky.size), 1, 16)
    G_full = jnp.asarray(
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape),
        dtype=jnp.complex128,
    )
    out_full, _phi = integrate_linear(
        G_full, grid_full, geom, params, dt=0.05, steps=1, method="euler", terms=terms
    )
    for iky in (1, 4):
        grid = select_ky_grid(grid_full, iky)
        out, _phi = integrate_linear(
            G_full[:, :, iky : iky + 1],
            grid,
            geom,
            params,
            dt=0.05,
            steps=1,
            method="euler",
            terms=terms,
        )
        np.testing.assert_allclose(
            np.asarray(out),
            np.asarray(out_full)[:, :, iky : iky + 1],
            rtol=1.0e-10,
            atol=1.0e-12,
        )


@pytest.mark.parametrize("rate", [None, 0.5])
def test_linear_integrator_applies_linked_end_damping_per_step(
    only_term_config, only_terms, spectral_grid, rate
):
    """Legacy linear damping scales as 1/dt; its Euler increment is fixed."""
    grid_full = spectral_grid(
        Nx=1,
        Ny=16,
        Nz=96,
        Lx=62.8,
        Ly=20.0 * np.pi,
        boundary="linked",
        y0=10.0,
        ntheta=32,
        nperiod=2,
    )
    ky_idx = int(np.argmin(np.abs(np.asarray(grid_full.ky) - 0.3)))
    grid = select_ky_grid(grid_full, ky_idx)
    geom = SAlphaGeometry.from_config(GeometryConfig(s_hat=0.8))
    params = LinearParams(
        damp_ends_amp=0.1, damp_ends_widthfrac=0.125, damp_ends_rate=rate
    )
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=4)
    G = jnp.ones((2, 4, 1, 1, 96), dtype=jnp.complex64)
    term_cfg = only_term_config(end_damping=1.0)

    _rhs_raw, _fields_raw, contrib_raw = assemble_rhs_terms_cached(
        G, cache, params, terms=term_cfg
    )
    _rhs_dt, _fields_dt, contrib_dt = assemble_rhs_terms_cached(
        G, cache, params, terms=term_cfg, dt=0.2
    )

    end_raw = np.asarray(contrib_raw["end_damping"])
    end_dt = np.asarray(contrib_dt["end_damping"])
    mask = np.abs(end_raw) > 1.0e-12
    assert np.any(mask)
    scale = 1.0 if rate is not None else 1 / 0.2
    assert np.allclose(end_dt[mask], end_raw[mask] * scale, rtol=1.0e-6, atol=1.0e-6)

    # Legacy increments are fixed; explicit-rate Euler increments scale with dt.
    terms = only_terms(end_damping=1.0)
    increments = []
    for dt in (0.1, 0.2):
        integrated, _phi = integrate_linear(
            G,
            grid,
            geom,
            params,
            dt=dt,
            steps=1,
            method="euler",
            terms=terms,
        )
        increment = np.asarray(integrated) - np.asarray(G)
        increments.append(increment / dt if rate is not None else increment)
    assert np.any(np.abs(increments[0][mask]) > 1.0e-12)
    assert np.allclose(increments[0], increments[1], rtol=1.0e-6, atol=1.0e-8)


@pytest.mark.parametrize(
    "method,order", [("euler", 1), ("rk2", 2), ("rk3", 3), ("rk4", 4)]
)
@pytest.mark.parametrize("dt", [0.002, 0.2])
@pytest.mark.parametrize("fixed_rate", [False, True])
def test_end_damping_rk_stability_polynomial_and_tangent(method, order, dt, fixed_rate):
    """An isolated damped scalar follows R(-A), not an exact removed fraction."""
    from math import factorial

    from gkx.solvers_time_explicit_steps import _linear_native_step
    from gkx.terms.assembly import _scalar_params

    with jax.enable_x64():
        strength = jnp.asarray(0.2, dtype=jnp.float64)

        def step(amplitude):
            rate = _scalar_params(
                LinearParams(damp_ends_rate=amplitude)
                if fixed_rate
                else LinearParams(damp_ends_amp=amplitude),
                jnp.float64,
                dt,
            ).damp_amp
            return _linear_native_step(
                jnp.asarray(1.0),
                jnp.asarray(0.0),
                jnp.asarray(dt),
                method_key=method,
                rhs=lambda value: -rate * value,
            )

        value, tangent = jax.jvp(step, (strength,), (jnp.ones_like(strength),))
        factor = dt if fixed_rate else 1.0
        expected = sum((-0.2 * factor) ** k / factorial(k) for k in range(order + 1))
        derivative = -factor * sum(
            (-0.2 * factor) ** k / factorial(k) for k in range(order)
        )
        assert float(value) == pytest.approx(expected, rel=0, abs=2e-15)
        assert float(tangent) == pytest.approx(derivative, rel=0, abs=2e-15)


@pytest.mark.parametrize("dt", [None, 0.0, 0.002, 0.2])
def test_fixed_end_damping_rate_pytree_and_reverse_derivative(dt):
    """Rate survives pytree/JIT and never depends on the caller's timestep."""

    def objective(params):
        return params.end_damping_strength(dt, jnp.float32) ** 2

    params = LinearParams(damp_ends_amp=99.0, damp_ends_rate=0.3)
    value, derivative = jax.jit(jax.value_and_grad(objective))(params)
    assert float(value) == pytest.approx(0.09)
    assert float(derivative.damp_ends_rate) == pytest.approx(0.6)
    assert float(derivative.damp_ends_amp) == 0.0


def test_streaming_zero_for_constant_z(cyclone_world, only_terms):
    """Streaming should vanish for z-constant fields."""
    cfg, grid, geom = cyclone_world(Nx=8, Ny=6, Nz=8, Lx=6.0, Ly=6.0)
    params = LinearParams(
        omega_d_scale=0.0,
        omega_star_scale=0.0,
        nu=0.0,
        nu_hyper=0.0,
        nu_hyper_l=0.0,
        nu_hyper_m=0.0,
        nu_hyper_lm=0.0,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
    )

    G = jnp.zeros((2, 3, grid.ky.size, cfg.grid.Nx, cfg.grid.Nz))
    G = G.at[:, 1:, ...].set(1.0)
    terms = only_terms(streaming=1.0)
    dG, _phi = linear_rhs(G, grid, geom, params, terms=terms)
    assert jnp.allclose(dG, 0.0)


def test_linear_rhs_shapes(cyclone_world):
    """RHS and potential should have consistent shapes."""
    cfg, grid, geom = cyclone_world(Nx=8, Ny=6, Nz=8, Lx=6.0, Ly=6.0)
    params = LinearParams()
    G = jnp.zeros((2, 3, grid.ky.size, cfg.grid.Nx, cfg.grid.Nz))
    dG, phi = linear_rhs(G, grid, geom, params)
    assert dG.shape == G.shape
    assert phi.shape == (grid.ky.size, cfg.grid.Nx, cfg.grid.Nz)


def test_linear_param_validation(cyclone_world):
    """Invalid parameters should be rejected in checked paths."""
    cfg, grid, geom = cyclone_world(Nx=8, Ny=6, Nz=8, Lx=6.0, Ly=6.0)
    G = jnp.zeros((2, 3, grid.ky.size, cfg.grid.Nx, cfg.grid.Nz))
    with pytest.raises(ValueError):
        compute_b(grid, geom, rho=0.0)
    with pytest.raises(ValueError):
        quasineutrality_phi(
            G[None, ...],
            jnp.ones((1, 2, grid.ky.size, cfg.grid.Nx, cfg.grid.Nz)),
            tau_e=-1.0,
            charge=jnp.array([1.0]),
            density=jnp.array([1.0]),
            tz=jnp.array([1.0]),
        )
    with pytest.raises(ValueError):
        streaming_term(G, dz=1.0, vth=0.0)
    with pytest.raises(ValueError):
        grad_z_periodic(G, dz=0.0)
    with pytest.raises(ValueError):
        linear_rhs(G.reshape(2, 3, -1), grid, geom, LinearParams())


def test_streaming_term_zero():
    """Zero fields should return zero streaming."""
    H = jnp.zeros((2, 3, 1, 1, 8))
    out = streaming_term(H, dz=1.0, vth=1.0)
    assert jnp.allclose(out, 0.0)


def test_integrate_linear_shapes(cyclone_world):
    """Integrator should return a time series of phi with expected length."""
    cfg, grid, geom = cyclone_world(Nx=8, Ny=6, Nz=8, Lx=6.0, Ly=6.0)
    params = LinearParams()
    G = jnp.zeros((2, 2, grid.ky.size, cfg.grid.Nx, cfg.grid.Nz))
    _, phi_t = integrate_linear(G, grid, geom, params, dt=0.1, steps=3, method="rk4")
    assert phi_t.shape[0] == 3


def test_integrate_linear_progress_with_sample_stride_gt_one(cyclone_world):
    """Sampled progress reporting must compute diagnostics before emitting callbacks."""
    cfg, grid, geom = cyclone_world(Nx=4, Ny=4, Nz=8, Lx=6.0, Ly=6.0)
    params = LinearParams()
    G = jnp.zeros((2, 2, grid.ky.size, cfg.grid.Nx, cfg.grid.Nz))

    _, phi_t = integrate_linear(
        G,
        grid,
        geom,
        params,
        dt=0.1,
        steps=4,
        method="rk2",
        sample_stride=2,
        show_progress=True,
    )

    assert phi_t.shape[0] == 2


def test_integrate_linear_methods(cyclone_world):
    """Explicit and IMEX paths should run without error."""
    cfg, grid, geom = cyclone_world(Nx=6, Ny=6, Nz=8, Lx=6.0, Ly=6.0)
    params = LinearParams()
    G = jnp.zeros((2, 2, grid.ky.size, cfg.grid.Nx, cfg.grid.Nz))
    for method in ("euler", "rk2", "imex", "semi-implicit", "sspx3"):
        _, phi_t = integrate_linear(
            G, grid, geom, params, dt=0.1, steps=2, method=method
        )
        assert phi_t.shape[0] == 2


def test_integrate_linear_with_cache(cyclone_world):
    """Integrate with a precomputed cache path."""
    cfg, grid, geom = cyclone_world(Nx=6, Ny=6, Nz=8, Lx=6.0, Ly=6.0)
    params = LinearParams()
    G = jnp.zeros((2, 2, grid.ky.size, cfg.grid.Nx, cfg.grid.Nz))
    cache = build_linear_cache(grid, geom, params, G.shape[0], G.shape[1])
    _, phi_t = integrate_linear(
        G, grid, geom, params, dt=0.1, steps=2, method="rk4", cache=cache
    )
    assert phi_t.shape[0] == 2


def test_integrate_linear_donation_matches_nondonated(cyclone_world):
    """Donating the initial state must not change the trajectory."""
    cfg, grid, geom = cyclone_world(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    params = LinearParams()
    shape = (2, 2, grid.ky.size, cfg.grid.Nx, cfg.grid.Nz)
    G = (
        jnp.arange(np.prod(shape), dtype=jnp.float32)
        .reshape(shape)
        .astype(jnp.complex64)
    )
    cache = build_linear_cache(grid, geom, params, shape[0], shape[1])

    state, phi = integrate_linear(
        G, grid, geom, params, dt=0.01, steps=2, method="rk2", cache=cache
    )
    state_donated, phi_donated = integrate_linear(
        jnp.array(G),
        grid,
        geom,
        params,
        dt=0.01,
        steps=2,
        method="rk2",
        cache=cache,
        donate=True,
    )

    np.testing.assert_allclose(state_donated, state, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(phi_donated, phi, rtol=1.0e-6, atol=1.0e-6)


def test_integrate_linear_checkpoint_runs(cyclone_world):
    """Checkpointed integration should run on a tiny grid."""
    cfg, grid, geom = cyclone_world(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    params = LinearParams()
    G = jnp.zeros((2, 2, grid.ky.size, cfg.grid.Nx, cfg.grid.Nz))
    _, phi_t = integrate_linear(
        G, grid, geom, params, dt=0.1, steps=2, method="rk4", checkpoint=True
    )
    assert phi_t.shape[0] == 2


def test_integrate_linear_invalid_method(cyclone_world):
    """Invalid integrator names should raise a ValueError."""
    cfg, grid, geom = cyclone_world(Nx=6, Ny=6, Nz=8, Lx=6.0, Ly=6.0)
    params = LinearParams()
    G = jnp.zeros((2, 2, grid.ky.size, cfg.grid.Nx, cfg.grid.Nz))
    with pytest.raises(ValueError):
        integrate_linear(G, grid, geom, params, dt=0.1, steps=2, method="rk5")


def test_linear_cache_matches_rhs(cyclone_world):
    """Cached RHS should match the direct RHS for the same inputs."""
    cfg, grid, geom = cyclone_world(Nx=6, Ny=6, Nz=8, Lx=6.0, Ly=6.0)
    params = LinearParams()
    G = jnp.zeros((2, 2, grid.ky.size, cfg.grid.Nx, cfg.grid.Nz))
    cache = build_linear_cache(grid, geom, params, G.shape[0], G.shape[1])
    dG0, phi0 = linear_rhs(G, grid, geom, params)
    dG1, phi1 = linear_rhs_cached(G, cache, params)
    assert jnp.allclose(dG0, dG1)
    assert jnp.allclose(phi0, phi1)


def test_linear_cached_rhs_replaces_builtin_collision_operator(
    cyclone_world, only_terms
):
    """A custom collision model replaces collisions but not other terms."""

    cfg, grid, geom = cyclone_world(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    params = LinearParams()
    G = jnp.ones((1, 2, grid.ky.size, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex64)
    cache = build_linear_cache(grid, geom, params, Nl=1, Nm=2)
    terms = only_terms(collisions=0.25)

    class DragCollision:
        def apply(self, context):
            return -3.0 * context.distribution

    dG, _ = linear_rhs_cached(
        G,
        cache,
        params,
        terms=terms,
        collision_operator=DragCollision(),
    )
    np.testing.assert_allclose(np.asarray(dG), -0.75, atol=1.0e-6)


def test_linear_cached_rhs_rejects_invalid_collision_shape(cyclone_world):
    cfg, grid, geom = cyclone_world(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    params = LinearParams()
    G = jnp.ones((1, 2, grid.ky.size, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex64)
    cache = build_linear_cache(grid, geom, params, Nl=1, Nm=2)

    class InvalidCollision:
        def apply(self, context):
            return context.distribution[..., 0]

    with pytest.raises(ValueError, match="same state shape"):
        linear_rhs_cached(
            G,
            cache,
            params,
            terms=LinearTerms(collisions=1.0),
            collision_operator=InvalidCollision(),
        )


def test_linear_integrator_applies_custom_collision_each_step(
    cyclone_world, only_terms
):
    cfg, grid, geom = cyclone_world(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    params = LinearParams()
    G0 = jnp.ones((1, 2, grid.ky.size, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex64)

    class DragCollision:
        def apply(self, context):
            return -3.0 * context.distribution

    terms = only_terms(collisions=0.25)
    G_final, _ = integrate_linear(
        G0,
        grid,
        geom,
        params,
        dt=0.1,
        steps=2,
        method="euler",
        terms=terms,
        collision_operator=DragCollision(),
    )
    np.testing.assert_allclose(np.asarray(G_final), 0.925**2, atol=1.0e-6)


def test_linear_cache_tree_roundtrip(cyclone_world):
    """LinearCache pytree should round-trip through flatten/unflatten."""
    cfg, grid, geom = cyclone_world(Nx=4, Ny=4, Nz=4, Lx=6.0, Ly=6.0)
    params = LinearParams()
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=2)
    children, aux = cache.tree_flatten()
    cache2 = LinearCache.tree_unflatten(aux, children)
    assert jnp.allclose(cache2.Jl, cache.Jl)
    assert cache2.kperp2_bmag is cache.kperp2_bmag
    assert jnp.allclose(cache2.omega_d, cache.omega_d)
    assert jnp.allclose(cache2.lb_lam, cache.lb_lam)
    assert jnp.allclose(cache2.hyper_ratio, cache.hyper_ratio)


def test_build_linear_cache_multispecies(cyclone_world):
    """Cache should support multiple species arrays."""
    cfg, grid, geom = cyclone_world(Nx=4, Ny=4, Nz=4, Lx=6.0, Ly=6.0)
    params = LinearParams(rho=jnp.array([1.0, 0.5]))
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=2)
    assert cache.Jl.shape[0] == 2


def test_linear_rhs_multispecies_shapes(cyclone_world):
    """Multispecies RHS should return a matching shape."""
    cfg, grid, geom = cyclone_world(Nx=4, Ny=4, Nz=4, Lx=6.0, Ly=6.0)
    params = LinearParams(
        charge_sign=jnp.array([1.0, -1.0]),
        density=jnp.array([1.0, 1.0]),
        mass=jnp.array([1.0, 0.001]),
        temp=jnp.array([1.0, 1.0]),
        vth=jnp.array([1.0, 1.0]),
        rho=jnp.array([1.0, 0.5]),
        tz=jnp.array([1.0, -1.0]),
        fprim=jnp.array([0.0, 0.0]),
        tprim=jnp.array([0.0, 0.0]),
    )
    G = jnp.zeros(
        (2, 2, 2, grid.ky.size, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex64
    )
    cache = build_linear_cache(grid, geom, params, Nl=G.shape[1], Nm=G.shape[2])
    dG, phi = linear_rhs_cached(G, cache, params)
    assert dG.shape == G.shape
    assert phi.shape == (grid.ky.size, cfg.grid.Nx, cfg.grid.Nz)


def test_implicit_preconditioner_hermite_line_shape_and_finite(cyclone_world):
    """Hermite-line preconditioner should run and preserve shape/dtype."""

    cfg, grid, geom = cyclone_world(Nx=2, Ny=4, Nz=16, Lx=62.8, Ly=62.8)
    params = LinearParams(
        fprim=cfg.model.fprim,
        tprim=cfg.model.tprim_i,
        omega_d_scale=0.2,
        omega_star_scale=0.55,
        rho_star=0.9,
        kpar_scale=float(geom.gradpar()),
    )
    Nl, Nm = 4, 6
    cache = build_linear_cache(grid, geom, params, Nl, Nm)
    base_dtype = jnp.complex128 if _x64_enabled() else jnp.complex64
    G0 = jnp.zeros(
        (1, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=base_dtype
    )
    _G, _shape, size, _dt_val, precond_op, _matvec, _squeeze = _build_implicit_operator(
        G0,
        cache,
        params,
        dt=0.1,
        terms=LinearTerms(),
        implicit_preconditioner="hermite-line",
    )
    x = jnp.ones((size,), dtype=base_dtype)
    y = precond_op(x)
    assert y.shape == (size,)
    assert jnp.all(jnp.isfinite(jnp.real(y)))
    assert jnp.all(jnp.isfinite(jnp.imag(y)))


def test_implicit_preconditioner_linked_hermite_line_coarse_shape_and_finite(
    cyclone_world, only_terms
):
    """Linked Hermite-line coarse preconditioner should preserve finite vectors."""

    cfg, grid, geom = cyclone_world(
        Nx=4, Ny=4, Nz=8, Lx=6.28, Ly=6.28, boundary="linked"
    )
    params = LinearParams(
        omega_d_scale=0.1,
        omega_star_scale=0.1,
        kpar_scale=float(geom.gradpar()),
    )
    Nl, Nm = 2, 4
    cache = build_linear_cache(grid, geom, params, Nl, Nm)
    base_dtype = jnp.complex128 if _x64_enabled() else jnp.complex64
    G0 = jnp.zeros(
        (1, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size),
        dtype=base_dtype,
    )
    _G, _shape, size, _dt_val, precond_op, matvec, _squeeze = _build_implicit_operator(
        G0,
        cache,
        params,
        dt=0.05,
        terms=LinearTerms(),
        implicit_preconditioner="hermite-line-coarse",
    )
    x = jnp.ones((size,), dtype=base_dtype)
    y = precond_op(x)
    z = matvec(x)
    assert y.shape == (size,)
    assert z.shape == (size,)
    assert jnp.all(jnp.isfinite(jnp.real(y)))
    assert jnp.all(jnp.isfinite(jnp.imag(y)))
    assert jnp.all(jnp.isfinite(jnp.real(z)))


@pytest.fixture
def streaming_shift_invert_setup(cyclone_world, only_terms):
    """The streaming-only eigenproblem both shift-invert cases start from.

    Returns ``(v0, cache, params, terms)`` for a 2x4x16 Cyclone flux tube with
    every drive, collision and damping term switched off, ``Nl, Nm = 4, 8``, and
    a ones vector in the solver's working complex dtype.
    """

    cfg, grid, geom = cyclone_world(Nx=2, Ny=4, Nz=16, Lx=6.28, Ly=6.28)
    params = LinearParams(
        omega_d_scale=0.0,
        omega_star_scale=0.0,
        nu=0.0,
        nu_hyper=0.0,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
        kpar_scale=float(geom.gradpar()),
    )
    Nl, Nm = 4, 8
    cache = build_linear_cache(grid, geom, params, Nl, Nm)
    base_dtype = jnp.complex128 if _x64_enabled() else jnp.complex64
    v0 = jnp.ones((Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=base_dtype)
    terms = only_terms(streaming=1.0)
    return v0, cache, params, terms


@pytest.mark.parametrize("kz_damping", [False, True])
def test_shift_invert_nearest_pair_passes_physical_outer_residual(
    streaming_shift_invert_setup,
    kz_damping,
):
    """Nearest-shift selection should admit a converged streaming Ritz pair."""

    v0, cache, params, terms = streaming_shift_invert_setup
    params = replace(
        params,
        hypercollisions_const=float(not kz_damping),
        hypercollisions_kz=float(kz_damping),
    )
    eigenvalue, eigenvector = dominant_eigenpair(
        v0,
        cache,
        params,
        terms,
        method="shift_invert",
        krylov_dim=4,
        restarts=1,
        shift=0.5j,
        shift_source="reference",
        shift_tol=1.0e-4,
        # The right-preconditioned solve is certified in the physical residual,
        # not the smaller transformed norm. This streaming case needs the
        # larger inner space to make that stronger contract portable.
        shift_maxiter=120,
        shift_restart=60,
        shift_solve_method="batched",
        shift_preconditioner="damping",
        shift_selection="nearest",
        shift_outer_residual_tol=0.06,
        mode_family="none",
        fallback_method="none",
    )
    assert jnp.isfinite(eigenvalue)
    assert jnp.all(jnp.isfinite(eigenvector))


def test_shift_invert_preconditioner_rejects_unconverged_outer_pair(
    streaming_shift_invert_setup,
):
    """A completed inner solve must not promote a large-residual Ritz pair."""

    v0, cache, params, terms = streaming_shift_invert_setup
    with pytest.raises(RuntimeError, match="failed the outer residual gate"):
        dominant_eigenpair(
            v0,
            cache,
            params,
            terms,
            method="shift_invert",
            krylov_dim=4,
            restarts=1,
            shift=0.5j,
            shift_tol=1.0e-2,
            shift_maxiter=20,
            shift_restart=10,
            shift_solve_method="batched",
            shift_preconditioner="hermite-line",
            fallback_method="none",
        )


def test_linear_cache_rho_star_scales_ky(cyclone_world):
    """rho_star should scale cached ky for normalization control."""
    cfg, grid, geom = cyclone_world(Nx=4, Ny=4, Nz=4, Lx=6.0, Ly=6.0)
    params = LinearParams(rho_star=2.0)
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=2)
    assert jnp.allclose(cache.ky, grid.ky * 2.0)


def test_linear_rhs_cached_invalid_shape(cyclone_world):
    """Cached RHS should reject invalid shapes."""
    cfg, grid, geom = cyclone_world(Nx=4, Ny=4, Nz=4, Lx=6.0, Ly=6.0)
    params = LinearParams()
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=2)
    with pytest.raises(ValueError):
        linear_rhs_cached(jnp.zeros((2, 3, 4)), cache, params)


def test_jit_path_handles_tracers(cyclone_world):
    """JIT tracing should exercise the tracer-safe validation path."""
    import jax

    cfg, grid, geom = cyclone_world(Nx=8, Ny=6, Nz=8, Lx=6.0, Ly=6.0)
    params = LinearParams()
    G = jnp.zeros((2, 2, grid.ky.size, cfg.grid.Nx, cfg.grid.Nz))

    @jax.jit
    def _run(G_in):
        return linear_rhs(G_in, grid, geom, params)[0]

    out = _run(G)
    assert out.shape == G.shape


def test_integrate_linear_implicit_runs(cyclone_world):
    """Implicit path should run on a tiny grid."""
    cfg, grid, geom = cyclone_world(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    params = LinearParams(nu=0.1)
    G = jnp.zeros((1, 1, grid.ky.size, cfg.grid.Nx, cfg.grid.Nz))
    _, phi_t = integrate_linear(
        G,
        grid,
        geom,
        params,
        dt=0.1,
        steps=1,
        method="implicit",
        implicit_iters=2,
        implicit_relax=0.5,
    )
    assert phi_t.shape[0] == 1


def test_implicit_standard_and_diagnostic_routes_match(cyclone_world, only_terms):
    """Implicit diagnostics must use the same prepared solve step."""

    cfg, grid, geom = cyclone_world(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    params = LinearParams(nu=0.1)
    state = jnp.zeros((1, 1, grid.ky.size, cfg.grid.Nx, cfg.grid.Nz))
    terms = only_terms()
    common = dict(
        dt=0.1,
        steps=2,
        method="implicit",
        terms=terms,
        sample_stride=1,
        implicit_iters=0,
        implicit_maxiter=1,
        implicit_restart=2,
    )

    standard, fields = integrate_linear(state, grid, geom, params, **common)
    diagnostic, phi, density = integrate_linear_diagnostics(
        state, grid, geom, params, **common
    )

    np.testing.assert_array_equal(np.asarray(diagnostic), np.asarray(standard))
    np.testing.assert_allclose(np.asarray(phi), np.asarray(fields))
    assert density.shape[0] == 2


def test_apply_hermite_v_simple():
    """Hermite v operator should map a single mode to neighbors."""
    G = jnp.zeros((1, 3, 1, 1, 1))
    G = G.at[0, 1, 0, 0, 0].set(1.0)
    out = apply_hermite_v(G)
    assert jnp.isclose(out[0, 0, 0, 0, 0], 1.0)
    assert jnp.isclose(out[0, 2, 0, 0, 0], jnp.sqrt(2.0))


def test_apply_laguerre_x_simple():
    """Laguerre x operator should reproduce the three-term recurrence."""
    G = jnp.zeros((3, 1, 1, 1, 1))
    G = G.at[1, 0, 0, 0, 0].set(1.0)
    out = apply_laguerre_x(G)
    assert jnp.isclose(out[0, 0, 0, 0, 0], -1.0)
    assert jnp.isclose(out[1, 0, 0, 0, 0], 3.0)
    assert jnp.isclose(out[2, 0, 0, 0, 0], -2.0)


def test_energy_operator_and_drive_coeffs():
    """Energy and drive coefficient helpers should return consistent shapes."""
    G = jnp.zeros((2, 3, 1, 1, 1))
    energy = energy_operator(G, coeff_const=1.0, coeff_par=0.5, coeff_perp=1.0)
    assert energy.shape == G.shape
    coeffs = diamagnetic_drive_coeffs(
        2, 3, eta_i=jnp.array(0.0), coeff_const=1.0, coeff_par=0.5, coeff_perp=1.0
    )
    assert coeffs.shape == (2, 3)
    assert jnp.isclose(coeffs[0, 0], 1.0)
    assert jnp.allclose(coeffs[1:, :], 0.0)


def test_mirror_curvature_terms_activate_with_drift_scale(cyclone_world, only_terms):
    """Drift/mirror terms should activate when omega_d_scale is nonzero."""
    cfg, grid, geom = cyclone_world(Nx=2, Ny=2, Nz=8, Lx=6.0, Ly=6.0)
    G = jnp.zeros((1, 3, grid.ky.size, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex64)
    G = G.at[0, 1, 1, 0, :].set(1.0 + 0.0j)

    params_off = LinearParams(
        omega_d_scale=0.0,
        omega_star_scale=0.0,
        kpar_scale=0.0,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
    )
    cache_off = build_linear_cache(grid, geom, params_off, G.shape[0], G.shape[1])
    terms_off = only_terms(bpar=1.0)
    dG_off, _phi_off = linear_rhs_cached(G, cache_off, params_off, terms=terms_off)
    assert jnp.allclose(dG_off, 0.0)

    params_on = LinearParams(
        omega_d_scale=1.0,
        omega_star_scale=0.0,
        kpar_scale=0.0,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
    )
    cache_on = build_linear_cache(grid, geom, params_on, G.shape[0], G.shape[1])
    terms_on = only_terms(mirror=1.0, curvature=1.0, gradb=1.0, bpar=1.0)
    dG_on, _phi_on = linear_rhs_cached(G, cache_on, params_on, terms=terms_on)
    assert jnp.max(jnp.abs(dG_on)) > 0.0


def test_diamagnetic_drive_populates_second_hermite_moment(cyclone_world, only_terms):
    """Diamagnetic drive should populate the m=2 component when enabled."""
    cfg, grid, geom = cyclone_world(Nx=2, Ny=4, Nz=8, Lx=6.0, Ly=6.0)
    G = jnp.zeros((2, 3, grid.ky.size, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex64)
    ky_index = 1
    G = G.at[0, 0, ky_index, 0, :].set(1.0 + 0.0j)

    params_off = LinearParams(
        omega_d_scale=0.0,
        omega_star_scale=0.0,
        kpar_scale=0.0,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
    )
    cache_off = build_linear_cache(grid, geom, params_off, G.shape[0], G.shape[1])
    terms_off = only_terms(bpar=1.0)
    dG_off, _phi_off = linear_rhs_cached(G, cache_off, params_off, terms=terms_off)
    assert jnp.allclose(dG_off[:, 2, ...], 0.0)

    params_on = LinearParams(
        omega_d_scale=0.0,
        omega_star_scale=1.0,
        kpar_scale=0.0,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
    )
    cache_on = build_linear_cache(grid, geom, params_on, G.shape[0], G.shape[1])
    terms_on = only_terms(diamagnetic=1.0, bpar=1.0)
    dG_on, _phi_on = linear_rhs_cached(G, cache_on, params_on, terms=terms_on)
    assert jnp.max(jnp.abs(dG_on[:, 2, ...])) > 0.0
    assert jnp.allclose(dG_on[:, 1, ...], 0.0)


def test_diamagnetic_drive_vanishes_for_zonal_mode(cyclone_world, only_terms):
    """Diamagnetic drive should vanish for the ky=0 mode."""
    cfg, grid, geom = cyclone_world(Nx=2, Ny=4, Nz=8, Lx=6.0, Ly=6.0)
    G = jnp.zeros((2, 3, grid.ky.size, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex64)
    G = G.at[0, 0, 0, 0, :].set(1.0 + 0.0j)
    params = LinearParams(
        omega_d_scale=0.0,
        omega_star_scale=1.0,
        kpar_scale=0.0,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
    )
    cache = build_linear_cache(grid, geom, params, G.shape[0], G.shape[1])
    terms = only_terms(diamagnetic=1.0, bpar=1.0)
    dG, _phi = linear_rhs_cached(G, cache, params, terms=terms)
    assert jnp.allclose(dG, 0.0)


def test_rho_star_scales_cache_ky(cyclone_world):
    """rho_star should scale the cached ky values."""
    cfg, grid, geom = cyclone_world(Nx=4, Ny=4, Nz=8, Lx=6.0, Ly=6.0)
    params = LinearParams(rho_star=2.0)
    cache = build_linear_cache(grid, geom, params, Nl=1, Nm=1)
    assert jnp.allclose(cache.ky, 2.0 * grid.ky)


def test_shift_axis_noop():
    """shift_axis should return the input when offset is zero."""
    from gkx.operators.linear.moments import shift_axis

    arr = jnp.arange(6.0).reshape(2, 3)
    out = shift_axis(arr, 0, axis=0)
    assert jnp.allclose(out, arr)


def test_apar_streaming_coupling_changes_rhs(cyclone_world):
    """Finite beta should modify streaming via Apar coupling."""
    cfg, grid, geom = cyclone_world(Nx=1, Ny=4, Nz=8, Lx=6.0, Ly=6.0)

    G = jnp.zeros((2, 3, grid.ky.size, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex64)
    z = grid.z
    G = G.at[0, 1, 1, 0, :].set(jnp.sin(z) + 0.0j)

    params_base = LinearParams(
        kpar_scale=1.0,
        omega_d_scale=0.0,
        omega_star_scale=0.0,
        nu=0.0,
        nu_hyper=0.0,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
        beta=0.0,
        fapar=0.0,
    )
    params_beta = LinearParams(
        kpar_scale=1.0,
        omega_d_scale=0.0,
        omega_star_scale=0.0,
        nu=0.0,
        nu_hyper=0.0,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
        beta=1.0,
        fapar=1.0,
    )
    cache = build_linear_cache(grid, geom, params_beta, G.shape[0], G.shape[1])
    dG0, _phi0 = linear_rhs_cached(G, cache, params_base, terms=LinearTerms())
    dG1, _phi1 = linear_rhs_cached(G, cache, params_beta, terms=LinearTerms())
    assert jnp.max(jnp.abs(dG1 - dG0)) > 0.0


def test_linked_boundary_growth_gradient_matches_finite_difference() -> None:
    """The sheared flux-tube boundary differentiates, not only the periodic one.

    Building the cache inside a trace used to fail closed on twist-shift
    boundaries, so every gradient test had to fall back to ``periodic``. The
    link topology is integer and stays on the host; only the drive is traced.
    """

    cfg = CycloneBaseCase(
        grid=GridConfig(Nx=4, Ny=8, Nz=16, Lx=6.0, Ly=6.0, boundary="linked")
    )
    grid = build_spectral_grid(cfg.grid)
    geom = sample_flux_tube_geometry(SAlphaGeometry.from_config(cfg.geometry), grid.z)
    base_params = LinearParams()
    Nl, Nm = 2, 4
    tprim = jnp.asarray(6.9)

    cache = build_linear_cache(grid, geom, replace(base_params, tprim=tprim), Nl, Nm)
    assert bool(cache.use_twist_shift)
    assert int(cache.jtwist) != 0

    profile = 1.0e-3 * (1.0 + 0.2 * jnp.cos(grid.z))
    G0 = jnp.zeros(
        (Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64
    )
    G0 = G0.at[0, 0, 1, 0, :].set(profile * (1.0 + 0.35j * jnp.sin(grid.z)))

    dt, steps = 0.05, 20

    def growth(drive: jnp.ndarray) -> jnp.ndarray:
        params = replace(base_params, tprim=drive)
        traj, _fields = integrate_linear(
            G0, grid, geom, params, dt, steps, sample_stride=steps
        )
        energy = jnp.real(jnp.vdot(traj[-1], traj[-1]))
        return 0.5 * jnp.log(energy) / (dt * steps)

    forward = jax.jit(growth)
    gradient = jax.jit(jax.grad(growth))(tprim)
    step = jnp.asarray(0.05)
    centered_fd = (forward(tprim + step) - forward(tprim - step)) / (2.0 * step)

    assert bool(jnp.isfinite(gradient))
    assert float(gradient) != 0.0
    np.testing.assert_allclose(
        np.asarray(gradient), np.asarray(centered_fd), rtol=2.0e-2, atol=1.0e-6
    )


# ---- from test_linear_helpers_extra.py ----


def test_species_parameter_order_has_one_sharding_contract() -> None:
    assert _SPECIES_PARAM_NAMES == (
        "charge_sign",
        "density",
        "mass",
        "temp",
        "vth",
        "rho",
        "fprim",
        "tprim",
        "tprim_e",
        "nu",
        "tz",
    )


def test_linear_validation_helpers_scalar_and_array() -> None:
    _check_positive(1.0, "x")
    _check_nonnegative(0.0, "x")
    _check_positive(jnp.asarray([1.0, 2.0]), "arr")
    _check_nonnegative(jnp.asarray([0.0, 2.0]), "arr")

    with pytest.raises(ValueError):
        _check_positive(0.0, "x")
    with pytest.raises(ValueError):
        _check_nonnegative(-1.0, "x")
    with pytest.raises(ValueError):
        _check_positive(jnp.asarray([1.0, 0.0]), "arr")
    with pytest.raises(ValueError):
        _check_nonnegative(jnp.asarray([0.0, -1.0]), "arr")


def test_linear_validation_helpers_accept_traced_values() -> None:
    @jax.jit
    def _checked(x):
        _check_positive(x, "x")
        _check_nonnegative(x, "x")
        return x

    assert float(_checked(jnp.asarray(1.0))) == pytest.approx(1.0)


def test_linear_validation_helpers_still_check_host_values_inside_a_trace() -> None:
    """A concrete argument is validated under ``jit``, not waved through.

    The guard used to build a ``jnp`` copy of its argument and ask whether
    *that* was traced. Inside a trace it always is, so the answer was yes for
    every argument and the operator stopped validating anything at all under
    ``jit`` -- a negative ``vth`` or a zero ``dz`` written into a jitted run
    went straight into the kernels. Only a genuinely traced value may skip.
    """

    # The guard is called from inside a live trace and its verdict is carried
    # out rather than raised out, so the trace this test opens is also the
    # trace it closes.
    seen: dict[str, str | None] = {}

    def verdict(fn, value, name):
        try:
            fn(value, name)
        except ValueError as exc:
            return str(exc)
        return None

    def probe(x):
        seen["negative_scalar"] = verdict(_check_positive, -1.0, "vth")
        seen["zero_in_array"] = verdict(_check_positive, np.asarray([1.0, 0.0]), "dz")
        seen["negative_nonneg"] = verdict(_check_nonnegative, -1.0, "tau_e")
        seen["good_scalar"] = verdict(_check_positive, 1.0, "vth")
        seen["traced"] = verdict(_check_positive, x, "vth")
        return x * 2.0

    jax.jit(probe)(jnp.asarray(-1.0))
    assert seen["negative_scalar"] == "vth must be > 0"
    assert seen["zero_in_array"] == "dz must be > 0"
    assert seen["negative_nonneg"] == "tau_e must be >= 0"
    assert seen["good_scalar"] is None
    # A negative value that is genuinely traced still goes unchecked: its sign
    # is not known until the trace runs, which is the one case the guard may
    # skip.
    assert seen["traced"] is None


def test_as_species_array_and_preconditioner_resolution() -> None:
    np.testing.assert_allclose(
        np.asarray(_as_species_array(2.0, 3, "nu")), [2.0, 2.0, 2.0]
    )
    np.testing.assert_allclose(
        np.asarray(_as_species_array(jnp.asarray([1.0, 2.0]), 2, "nu")), [1.0, 2.0]
    )
    with pytest.raises(ValueError):
        _as_species_array(jnp.asarray([1.0, 2.0]), 3, "nu")

    assert _resolve_implicit_preconditioner(None) == "auto"
    assert _resolve_implicit_preconditioner("  Damping ") == "damping"

    def fn(x):
        return x

    assert _resolve_implicit_preconditioner(fn) is fn


def test_structured_tridiagonal_last_axis_matches_fused_reference_and_jvp() -> None:
    dtype = jnp.complex128 if jax.config.read("jax_enable_x64") else jnp.complex64
    real_dtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
    shape = (2, 3, 7)
    phase = jnp.arange(np.prod(shape), dtype=real_dtype).reshape(shape)
    lower = (0.03 + 0.01j) * jnp.cos(phase).astype(dtype)
    upper = (-0.02 + 0.015j) * jnp.sin(phase + 0.2).astype(dtype)
    lower = lower.at[..., 0].set(0.0)
    upper = upper.at[..., -1].set(0.0)
    diagonal = jnp.full(shape, 2.5 + 0.1j, dtype=dtype)
    rhs = (jnp.cos(phase) + 1j * jnp.sin(0.3 * phase)).astype(dtype)

    solve = jax.jit(
        lambda value: linear_implicit._solve_tridiagonal_last_axis(
            lower, diagonal, upper, value
        )
    )
    actual = solve(rhs)
    reference = jax.lax.linalg.tridiagonal_solve(
        lower, diagonal, upper, rhs[..., None]
    )[..., 0]
    tolerance = 2.0e-6 if dtype == jnp.complex64 else 1.0e-12
    np.testing.assert_allclose(actual, reference, rtol=tolerance, atol=tolerance)

    tangent = jnp.full(shape, 0.2 - 0.1j, dtype=dtype)
    _, tangent_out = jax.jvp(solve, (rhs,), (tangent,))
    np.testing.assert_allclose(
        tangent_out,
        solve(tangent),
        rtol=tolerance,
        atol=tolerance,
    )


def test_linear_dissipation_terms_have_single_canonical_owner() -> None:
    for name in linear_dissipation.__all__:
        assert getattr(linear_terms, name) is getattr(linear_dissipation, name)
        assert inspect.getmodule(getattr(linear_terms, name)) is linear_dissipation


def test_is_tracer_and_lenard_bernstein_eigenvalues() -> None:
    assert _is_tracer(1.0) is False
    traced_flag = jax.jit(
        lambda x: jnp.asarray(1 if _is_tracer(x) else 0, dtype=jnp.int32)
    )(1.0)
    assert int(traced_flag) == 1

    expected = np.asarray([[0.0, 0.3, 0.6], [0.7, 1.0, 1.3]], dtype=np.float32)
    got = np.asarray(
        lenard_bernstein_eigenvalues(2, 3, nu_hermite=0.3, nu_laguerre=0.7),
        dtype=np.float32,
    )
    np.testing.assert_allclose(got, expected)


def test_low_rank_moment_and_damping_cache_match_expected_shapes_and_values() -> None:
    params = LinearParams(
        nu_hermite=0.3, nu_laguerre=0.7, p_hyper=2, p_hyper_l=3, p_hyper_m=4
    )
    cache = _build_low_rank_moment_cache_arrays(2, 3, params, jnp.float32)

    expected_lb = np.asarray([[0.0, 0.3, 0.6], [0.7, 1.0, 1.3]], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(cache["lb_lam"]), expected_lb, rtol=1e-6)
    assert cache["lb_lam"].shape == (2, 3)
    assert cache["hyper_ratio"].shape == (2, 3, 1, 1, 1)
    assert cache["sqrt_p"].shape == (1, 1, 3, 1, 1, 1)
    assert cache["mask_const"].dtype == jnp.bool_

    periodic = np.asarray(
        _build_end_damping_profile_array(8, 0.25, "periodic", jnp.float32)
    )
    linked = np.asarray(
        _build_end_damping_profile_array(8, 0.25, "linked", jnp.float32)
    )
    np.testing.assert_allclose(periodic, np.zeros(8, dtype=np.float32))
    assert linked[0] > 0.0
    assert linked[-1] > 0.0


def test_low_rank_moment_cache_keeps_high_order_kz_hypercollision_finite() -> None:
    params = LinearParams(p_hyper_m=20.0)
    cache = _build_low_rank_moment_cache_arrays(24, 128, params, jnp.float32)

    assert np.all(np.isfinite(np.asarray(cache["m_pow"])))
    assert np.isfinite(float(np.asarray(cache["m_norm_kz_factor"])))
    assert float(np.max(np.asarray(cache["m_pow"]))) <= 1.0


def test_hypercollision_damping_preserves_low_moments_and_grows_with_kz() -> None:
    params = LinearParams(
        nu_hyper=0.0,
        nu_hyper_l=0.2,
        nu_hyper_m=0.3,
        nu_hyper_lm=0.4,
        hypercollisions_const=1.0,
        hypercollisions_kz=1.0,
        p_hyper_l=2.0,
        p_hyper_m=4.0,
        p_hyper_lm=2.0,
        vth=1.5,
        kpar_scale=2.0,
    )
    moment_cache = _build_low_rank_moment_cache_arrays(3, 5, params, jnp.float32)
    cache = SimpleNamespace(
        **moment_cache,
        kz=jnp.asarray([0.0, 1.0, 2.0], dtype=jnp.float32),
    )

    damping = np.asarray(hypercollision_damping(cache, params, jnp.float32))

    assert damping.shape == (1, 3, 5, 1, 1, 3)
    np.testing.assert_allclose(damping[0, 0, 0, 0, 0], 0.0, atol=0.0)
    np.testing.assert_allclose(damping[0, 1, 2, 0, 0], 0.0, atol=0.0)
    assert damping[0, 2, 0, 0, 0, 0] > 0.0
    assert damping[0, 0, 4, 0, 0, 2] > damping[0, 0, 4, 0, 0, 1]
    assert damping[0, 0, 4, 0, 0, 1] > damping[0, 0, 4, 0, 0, 0]


def test_gyroaverage_cache_helper_matches_species_vmap_convention() -> None:
    b = jnp.asarray(
        [
            [[[0.0, 0.2], [0.4, 0.6]]],
            [[[0.1, 0.3], [0.5, 0.7]]],
        ],
        dtype=jnp.float32,
    )
    Jl, JlB = _build_gyroaverage_cache_arrays(b, Nl=3, real_dtype=jnp.float32)
    expected = jax.vmap(lambda bs: J_l_all(bs, l_max=2))(b).astype(jnp.float32)

    np.testing.assert_allclose(np.asarray(Jl), np.asarray(expected), rtol=1e-6)
    assert Jl.shape == (2, 3, 1, 2, 2)
    assert JlB.shape == Jl.shape
    np.testing.assert_allclose(np.asarray(JlB[:, 0]), np.asarray(Jl[:, 0]), rtol=1e-6)


def test_linear_params_and_terms_roundtrip() -> None:
    params = LinearParams(
        charge_sign=jnp.asarray([1.0, -1.0]), nu=jnp.asarray([0.1, 0.2]), beta=0.3
    )
    leaves, treedef = jax.tree_util.tree_flatten(params)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    np.testing.assert_allclose(np.asarray(restored.charge_sign), [1.0, -1.0])
    np.testing.assert_allclose(np.asarray(restored.nu), [0.1, 0.2])
    assert float(restored.beta) == pytest.approx(0.3)

    terms = LinearTerms(apar=0.0, bpar=0.0, hyperdiffusion=1.0)
    term_cfg = linear_terms_to_term_config(terms, nonlinear=0.25)
    assert float(term_cfg.nonlinear) == pytest.approx(0.25)
    assert term_config_to_linear_terms(term_cfg) == terms

    assert linear_terms_to_term_config(None) == TermConfig()
    assert term_config_to_linear_terms(None) == LinearTerms()

    custom_cfg = TermConfig(
        streaming=0.2,
        mirror=0.3,
        curvature=0.4,
        gradb=0.5,
        diamagnetic=0.6,
        collisions=0.7,
        hypercollisions=0.8,
        hyperdiffusion=0.9,
        end_damping=0.1,
        apar=0.0,
        bpar=1.0,
        nonlinear=3.0,
    )
    assert term_config_to_linear_terms(custom_cfg) == LinearTerms(
        streaming=0.2,
        mirror=0.3,
        curvature=0.4,
        gradb=0.5,
        diamagnetic=0.6,
        collisions=0.7,
        hypercollisions=0.8,
        hyperdiffusion=0.9,
        end_damping=0.1,
        apar=0.0,
        bpar=1.0,
    )


def test_linear_term_classifiers_and_parallel_device_validation(only_terms) -> None:
    streaming_only = only_terms(streaming=1.0)
    electrostatic_slices = only_terms(
        streaming=1.0, mirror=1.0, curvature=1.0, gradb=1.0, diamagnetic=1.0
    )

    assert _is_streaming_only_terms(streaming_only) is True
    assert _is_streaming_only_terms(LinearTerms()) is False
    assert _is_electrostatic_slice_terms(electrostatic_slices) is True
    assert _is_electrostatic_slice_terms(LinearTerms(collisions=1.0)) is False
    assert _is_electrostatic_field_terms(LinearTerms(apar=0.0, bpar=0.0)) is True
    assert _is_electrostatic_field_terms(LinearTerms(apar=1.0, bpar=0.0)) is False

    devices = ["cpu0", "cpu1"]
    assert _resolve_parallel_devices(devices=devices) == devices
    assert _resolve_parallel_devices(devices=devices, num_devices=2) == devices
    with pytest.raises(ValueError, match="must match"):
        _resolve_parallel_devices(devices=devices, num_devices=1)
    with pytest.raises(ValueError, match="at least one"):
        _resolve_parallel_devices(devices=[])
    with pytest.raises(ValueError, match=">= 1"):
        _resolve_parallel_devices(num_devices=0)
    with pytest.raises(ValueError, match="requested"):
        _resolve_parallel_devices(num_devices=len(jax.devices()) + 1)


def test_signed_to_index_and_linked_end_damping_profile() -> None:
    assert _signed_to_index(0, 3) == 0
    assert _signed_to_index(1, 3) == 1
    assert _signed_to_index(-1, 3) == 2
    assert _signed_to_index(-3, 3) == -1

    linked = (jnp.asarray([[1, 4]], dtype=jnp.int32),)
    profile = _build_linked_end_damping_profile(
        linked_indices=linked,
        ny=3,
        nx=2,
        nz=4,
        widthfrac=0.5,
        ky_mode=np.asarray([0, 1, -1], dtype=np.int32),
    )
    assert profile.shape == (3, 2, 4)
    assert np.max(profile) > 0.0
    assert np.all(profile[0] == 0.0)

    empty = _build_linked_end_damping_profile(
        linked_indices=(),
        ny=2,
        nx=2,
        nz=2,
        widthfrac=0.5,
    )
    assert np.allclose(empty, 0.0)

    with pytest.raises(ValueError):
        _build_linked_end_damping_profile(
            linked_indices=linked,
            ny=3,
            nx=2,
            nz=4,
            widthfrac=0.5,
            ky_mode=np.asarray([0, 1], dtype=np.int32),
        )

    profile_zero_width = _build_linked_end_damping_profile(
        linked_indices=(jnp.asarray([[1]], dtype=jnp.int32),),
        ny=2,
        nx=1,
        nz=4,
        widthfrac=0.01,
    )
    assert np.allclose(profile_zero_width, 0.0)


@pytest.mark.parametrize("nz", [3, 4])
def test_linked_fft_maps_validate_ky_mode_and_empty_maps(nz) -> None:
    kwargs = dict(y0=1.0, nz=nz, dz=0.5, jtwist=1, real_dtype=jnp.float32)
    empty_indices, empty_kz = _build_linked_fft_maps(
        ky=np.asarray([]), kx=np.asarray([]), **kwargs
    )
    assert empty_indices == ()
    assert empty_kz == ()

    indices, kz = _build_linked_fft_maps(
        ky=np.asarray([0.0, 0.1, 0.2]),
        kx=np.asarray([0.0, 0.2, -0.2, 0.4]),
        **kwargs,
        ky_mode=np.asarray([0, 1, 2]),
    )
    assert [idx.tolist() for idx in indices] == [
        [[0], [3], [9], [2]],
        [[5, 11]],
        [[4, 1, 10]],
    ]
    assert len(kz) == 3
    for chain, frequencies in zip(indices, kz):
        n = chain.shape[1] * nz
        modes = np.r_[np.arange((n + 1) // 2), np.arange(-(n // 2), 0)]
        np.testing.assert_allclose(frequencies, 4 * np.pi * modes / n, rtol=1e-6)

    profile = _build_linked_end_damping_profile(
        linked_indices=(jnp.asarray([1, 2], dtype=jnp.int32),),
        ny=3,
        nx=2,
        nz=4,
        widthfrac=0.5,
    )
    assert np.allclose(profile, 0.0)


def test_build_linear_cache_linked_non_twist_contract(spectral_grid) -> None:
    grid = spectral_grid(
        Nx=4,
        Ny=4,
        Nz=8,
        Lx=2.0 * np.pi,
        Ly=2.0 * np.pi,
        y0=1.0,
        boundary="linked",
        jtwist=1,
        non_twist=True,
    )
    geom = SAlphaGeometry(q=1.4, s_hat=1.0, epsilon=0.1)
    params = LinearParams(nu_hyper=0.0, nu_hyper_m=0.0, damp_ends_widthfrac=0.25)

    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=3)

    assert cache.use_twist_shift is True
    assert cache.jtwist == 1
    assert cache.kperp2.shape == (grid.ky.size, grid.kx.size, grid.z.size)
    assert len(cache.linked_indices) == len(cache.linked_kz)
    assert cache.linked_damp_profile.shape == (grid.ky.size, grid.kx.size, grid.z.size)
    assert np.all(np.isfinite(np.asarray(cache.kperp2)))
    if cache.linked_indices:
        assert cache.linked_use_gather is True
        assert cache.linked_gather_map.shape == (grid.ky.size * grid.kx.size,)


def test_build_linear_cache_y0_default_and_zero_twist_branches(spectral_grid) -> None:
    grid = spectral_grid(
        Nx=4,
        Ny=4,
        Nz=8,
        Lx=2.0 * np.pi,
        Ly=2.0 * np.pi,
        boundary="linked",
        jtwist=None,
    )
    params = LinearParams(nu_hyper=0.0, nu_hyper_m=0.0, damp_ends_widthfrac=0.0)
    geom = SAlphaGeometry(q=1.4, s_hat=0.0, epsilon=0.1)

    cache = build_linear_cache(replace(grid, y0=None), geom, params, Nl=2, Nm=2)

    assert cache.use_twist_shift is True
    assert cache.jtwist == 1
    assert cache.linked_damp_profile.shape == (grid.ky.size, grid.kx.size, grid.z.size)
    assert np.allclose(np.asarray(cache.linked_damp_profile), 0.0)
    assert np.all(np.isfinite(np.asarray(cache.kperp2)))


def _sampled_geometry_with_shear(
    theta: jnp.ndarray, s_hat: jnp.ndarray
) -> FluxTubeGeometryData:
    shear = s_hat * theta
    ones = jnp.ones_like(theta)
    zeros = jnp.zeros_like(theta)
    return FluxTubeGeometryData(
        theta=theta,
        gradpar_value=1.0,
        bmag_profile=ones,
        bgrad_profile=zeros,
        gds2_profile=1.0 + shear * shear,
        gds21_profile=-s_hat * shear,
        gds22_profile=s_hat * s_hat * ones,
        cv_profile=jnp.cos(theta) + shear * jnp.sin(theta),
        gb_profile=jnp.cos(theta) + shear * jnp.sin(theta),
        cv0_profile=-s_hat * jnp.sin(theta),
        gb0_profile=-s_hat * jnp.sin(theta),
        jacobian_profile=ones,
        grho_profile=ones,
        q=1.4,
        s_hat=s_hat,
        epsilon=0.1,
        R0=1.0,
        source_model="sampled-test",
    )


def test_build_linear_cache_allows_traced_shear_for_periodic_sampled_geometry(
    spectral_grid,
) -> None:
    grid = spectral_grid(
        Nx=2, Ny=4, Nz=4, Lx=2.0 * np.pi, Ly=2.0 * np.pi, boundary="periodic"
    )
    theta = jnp.asarray(grid.z, dtype=jnp.float32)
    params = LinearParams(nu_hyper=0.0, nu_hyper_m=0.0)

    def objective(s_hat: jnp.ndarray) -> jnp.ndarray:
        geom = _sampled_geometry_with_shear(theta, s_hat)
        cache = build_linear_cache(grid, geom, params, Nl=1, Nm=1)
        return jnp.sum(cache.kperp2)

    grad = jax.grad(objective)(jnp.asarray(0.8, dtype=jnp.float32))

    assert np.isfinite(float(grad))


def test_laguerre_bessel_factors_have_analytic_zero_limit_and_tangent() -> None:
    zero = jnp.asarray(0.0, dtype=jnp.float64)

    def factors(alpha2: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack(_gyro_bessel_factors(alpha2))

    value, tangent = jax.jvp(factors, (zero,), (jnp.ones_like(zero),))

    np.testing.assert_allclose(value, np.asarray([1.0, 0.5]), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        tangent, np.asarray([-0.25, -0.0625]), rtol=1.0e-12, atol=1.0e-12
    )


def test_laguerre_bessel_cache_has_finite_geometry_tangent_at_zero_mode(
    spectral_grid,
) -> None:
    grid = spectral_grid(
        Nx=2, Ny=4, Nz=4, Lx=2.0 * np.pi, Ly=2.0 * np.pi, boundary="periodic"
    )
    theta = jnp.asarray(grid.z, dtype=jnp.float64)
    params = LinearParams(nu_hyper=0.0, nu_hyper_m=0.0)

    def observable(s_hat: jnp.ndarray) -> jnp.ndarray:
        geom = _sampled_geometry_with_shear(theta, s_hat)
        cache = build_linear_cache(grid, geom, params, Nl=2, Nm=1)
        return jnp.sum(cache.laguerre_j0) + jnp.sum(cache.laguerre_j1_over_alpha)

    shear = jnp.asarray(0.8, dtype=jnp.float64)
    tangent = jax.jvp(observable, (shear,), (jnp.ones_like(shear),))[1]

    assert np.isfinite(float(tangent))


def test_build_linear_cache_periodic_non_twist_uses_geometry_shear(
    spectral_grid,
) -> None:
    grid = spectral_grid(
        Nx=2,
        Ny=4,
        Nz=8,
        Lx=2.0 * np.pi,
        Ly=2.0 * np.pi,
        boundary="periodic",
        non_twist=True,
    )
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.1)
    params = LinearParams(nu_hyper=0.0, nu_hyper_m=0.0)

    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=3)

    assert cache.use_twist_shift is False
    assert np.all(np.isfinite(np.asarray(cache.kperp2)))
    assert np.all(np.isfinite(np.asarray(cache.cv_d)))
    assert np.all(np.isfinite(np.asarray(cache.gb_d)))


def test_sheared_kx_cache_zero_shear_identity_and_tangent(spectral_grid) -> None:
    grid = spectral_grid(
        Nx=4,
        Ny=4,
        Nz=8,
        Lx=2.0 * np.pi,
        Ly=2.0 * np.pi,
        boundary="periodic",
    )
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.1)
    params = LinearParams(rho_star=1.0, nu_hyper=0.0, nu_hyper_m=0.0)
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=3)

    identity = linear_cache.update_linear_cache_for_sheared_kx(
        cache,
        grid,
        geom,
        params,
        cache.kx_grid,
    )
    for name in (
        "kx_grid",
        "kperp2",
        "cv_d",
        "gb_d",
        "omega_d",
        "b",
        "Jl",
        "JlB",
        "laguerre_j0",
        "laguerre_j1_over_alpha",
    ):
        np.testing.assert_allclose(
            np.asarray(getattr(identity, name)),
            np.asarray(getattr(cache, name)),
            rtol=2.0e-6,
            atol=2.0e-7,
        )

    # The ky extent is the grid's stored row count, not Ny: what this test
    # compares is one cache against another on the same grid, and a ramp of
    # whatever length that grid holds serves it either way.
    state_shape = (2, 3, int(grid.ky.size), int(grid.kx.size), int(grid.z.size))
    G = (
        jnp.arange(int(np.prod(state_shape)), dtype=jnp.float32).reshape(state_shape)
        / 1000.0
    ).astype(jnp.complex64)
    rhs_base, phi_base = linear_rhs_cached(G, cache, params, use_jit=False)
    rhs_identity, phi_identity = linear_rhs_cached(
        G,
        identity,
        params,
        use_jit=False,
    )
    np.testing.assert_allclose(rhs_identity, rhs_base, rtol=3.0e-6, atol=3.0e-7)
    np.testing.assert_allclose(phi_identity, phi_base, rtol=3.0e-6, atol=3.0e-7)

    def observable(shift: jnp.ndarray) -> jnp.ndarray:
        effective_kx = cache.kx_grid - shift * cache.ky_grid
        updated = linear_cache.update_linear_cache_for_sheared_kx(
            cache,
            grid,
            geom,
            params,
            effective_kx,
        )
        return jnp.sum(updated.kperp2) + 0.1 * jnp.sum(updated.Jl)

    shift = jnp.asarray(0.17, dtype=jnp.float32)
    tangent = jax.jvp(observable, (shift,), (jnp.ones_like(shift),))[1]
    step = jnp.asarray(5.0e-3, dtype=jnp.float32)
    finite_difference = (observable(shift + step) - observable(shift - step)) / (
        2.0 * step
    )
    np.testing.assert_allclose(tangent, finite_difference, rtol=5.0e-4)

    non_twist_grid = replace(grid, boundary="linked", non_twist=True)
    with pytest.raises(NotImplementedError, match="periodic or linked standard"):
        linear_cache.update_linear_cache_for_sheared_kx(
            cache,
            non_twist_grid,
            geom,
            params,
            cache.kx_grid,
        )
    with pytest.raises(ValueError, match="shape \(ky, kx\)"):
        linear_cache.update_linear_cache_for_sheared_kx(
            cache,
            grid,
            geom,
            params,
            cache.kx,
        )


def test_hyperdiffusion_accepts_sheared_two_dimensional_kx() -> None:
    G = jnp.ones((1, 1, 4, 4, 2), dtype=jnp.complex64)
    kx = jnp.asarray([0.0, 1.0, -2.0, -1.0], dtype=jnp.float32)
    ky = jnp.asarray([0.0, 1.0, -2.0, -1.0], dtype=jnp.float32)
    mask = jnp.ones((4, 4), dtype=bool)
    kwargs = {
        "ky": ky,
        "dealias_mask": mask,
        "D_hyper": jnp.asarray(0.1),
        "p_hyper_kperp": jnp.asarray(2.0),
        "weight": jnp.asarray(1.0),
    }
    one_dimensional = linear_dissipation.hyperdiffusion_contribution(
        G,
        kx=kx,
        **kwargs,
    )
    two_dimensional = linear_dissipation.hyperdiffusion_contribution(
        G,
        kx=jnp.broadcast_to(kx[None, :], mask.shape),
        **kwargs,
    )
    np.testing.assert_allclose(two_dimensional, one_dimensional)


def test_build_linear_cache_rejects_traced_shear_for_twist_shift_geometry(
    spectral_grid,
) -> None:
    grid = spectral_grid(
        Nx=2,
        Ny=4,
        Nz=4,
        Lx=2.0 * np.pi,
        Ly=2.0 * np.pi,
        boundary="linked",
        jtwist=1,
    )
    theta = jnp.asarray(grid.z, dtype=jnp.float32)
    params = LinearParams(nu_hyper=0.0, nu_hyper_m=0.0)

    def objective(s_hat: jnp.ndarray) -> jnp.ndarray:
        geom = _sampled_geometry_with_shear(theta, s_hat)
        cache = build_linear_cache(grid, geom, params, Nl=1, Nm=1)
        return jnp.sum(cache.kperp2)

    with pytest.raises(
        ValueError,
        match="differentiating with respect to magnetic shear is not supported",
    ):
        jax.grad(objective)(jnp.asarray(0.8, dtype=jnp.float32))


def test_build_H_field_couplings_and_errors() -> None:
    G5 = jnp.zeros((2, 2, 1, 1, 3), dtype=jnp.complex64)
    Jl4 = jnp.ones((2, 1, 1, 3), dtype=jnp.float32)
    phi = jnp.ones((1, 1, 3), dtype=jnp.complex64)
    apar = 0.5 * phi
    bpar = 0.25 * phi

    H = build_H(
        G5,
        Jl4,
        phi,
        tz=jnp.asarray(2.0),
        apar=apar,
        vth=jnp.asarray(3.0),
        bpar=bpar,
        JlB=Jl4,
    )

    assert H.shape == G5.shape
    assert np.max(np.abs(np.asarray(H[0, 0]))) > 0.0
    assert np.max(np.abs(np.asarray(H[0, 1]))) > 0.0
    with pytest.raises(ValueError, match="vth"):
        build_H(G5, Jl4, phi, tz=1.0, apar=apar)
    with pytest.raises(ValueError, match="JlB"):
        build_H(G5, Jl4, phi, tz=1.0, bpar=bpar)


def test_linear_rhs_rejects_invalid_state_rank() -> None:
    with pytest.raises(ValueError, match="G must have shape"):
        linear_rhs(
            jnp.zeros((2, 2), dtype=jnp.complex64), object(), object(), LinearParams()
        )


def test_linear_rhs_accepts_multispecies_state() -> None:
    grid = build_spectral_grid(
        GridConfig(Nx=2, Ny=2, Nz=4, Lx=2.0 * np.pi, Ly=2.0 * np.pi)
    )
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.1)
    params = LinearParams(charge_sign=jnp.asarray([1.0]), nu_hyper=0.0, nu_hyper_m=0.0)
    G = jnp.zeros(
        (1, 2, 2, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64
    )

    dG, phi = linear_rhs(G, grid, geom, params, terms=LinearTerms())

    assert dG.shape == G.shape
    assert phi.shape == (grid.ky.size, grid.kx.size, grid.z.size)
    assert np.all(np.isfinite(np.asarray(dG)))


def test_build_implicit_operator_handles_species_squeeze(monkeypatch) -> None:
    monkeypatch.setattr(
        "gkx.solvers_linear_implicit.hypercollision_kz_coefficient", lambda *args: 0.0
    )
    G0 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    cache = SimpleNamespace(
        lb_lam=jnp.ones((1, 2, 2, 1, 1, 2), dtype=jnp.float32),
        l=jnp.ones((2, 2, 1, 1, 2), dtype=jnp.float32),
        m=jnp.ones((2, 2, 1, 1, 2), dtype=jnp.float32),
        cv_d=jnp.ones((1, 1, 2), dtype=jnp.float32),
        gb_d=jnp.ones((1, 1, 2), dtype=jnp.float32),
        bgrad=jnp.ones((2,), dtype=jnp.float32),
        sqrt_m_ladder=jnp.ones((2,), dtype=jnp.float32),
        sqrt_p=jnp.ones((2,), dtype=jnp.float32),
        kz=jnp.array([0.0, 1.0], dtype=jnp.float32),
    )
    params = SimpleNamespace(
        nu=0.1,
        tz=1.0,
        vth=1.0,
        omega_d_scale=1.0,
        kpar_scale=1.0,
    )
    monkeypatch.setattr(
        "gkx.solvers_linear_implicit.hypercollision_damping",
        lambda cache, params, dtype: jnp.zeros_like(cache.lb_lam, dtype=dtype),
    )
    monkeypatch.setattr(
        "gkx.solvers_linear_implicit.linear_rhs_cached",
        lambda G, cache, params, **kwargs: (jnp.ones_like(G), None),
    )

    G, shape, size, dt_val, precond_op, matvec, squeeze_species = (
        _build_implicit_operator(
            G0,
            cache,
            params,
            dt=0.2,
            terms=LinearTerms(),
            implicit_preconditioner="damping",
        )
    )

    assert squeeze_species is True
    assert shape == (1, 2, 2, 1, 1, 2)
    assert size == 8
    assert G.shape == shape
    assert np.isfinite(np.asarray(precond_op(G))).all()
    assert np.isfinite(np.asarray(matvec(G))).all()
    assert float(dt_val) == pytest.approx(0.2)


def test_build_implicit_operator_preconditioner_aliases_and_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        "gkx.solvers_linear_implicit.hypercollision_kz_coefficient", lambda *args: 0.0
    )
    G0 = jnp.zeros((1, 2, 2, 1, 1, 2), dtype=jnp.complex64)
    cache = SimpleNamespace(
        lb_lam=jnp.ones((1, 2, 2, 1, 1, 2), dtype=jnp.float32),
        l=jnp.ones((1, 2, 2, 1, 1, 2), dtype=jnp.float32),
        m=jnp.ones((1, 2, 2, 1, 1, 2), dtype=jnp.float32),
        cv_d=jnp.ones((1, 1, 2), dtype=jnp.float32),
        gb_d=jnp.ones((1, 1, 2), dtype=jnp.float32),
        bgrad=jnp.ones((2,), dtype=jnp.float32),
        sqrt_m_ladder=jnp.ones((2,), dtype=jnp.float32),
        sqrt_p=jnp.ones((2,), dtype=jnp.float32),
        kz=jnp.array([0.0, 1.0], dtype=jnp.float32),
        linked_indices=(),
        linked_kz=(),
        use_twist_shift=False,
    )
    params = SimpleNamespace(
        nu=jnp.asarray([0.1], dtype=jnp.float32),
        tz=jnp.asarray([1.0], dtype=jnp.float32),
        vth=jnp.asarray([1.0], dtype=jnp.float32),
        omega_d_scale=1.0,
        kpar_scale=1.0,
    )
    monkeypatch.setattr(
        "gkx.solvers_linear_implicit.hypercollision_damping",
        lambda cache, params, dtype: jnp.zeros_like(cache.lb_lam, dtype=dtype),
    )
    monkeypatch.setattr(
        "gkx.solvers_linear_implicit.linear_rhs_cached",
        lambda G, cache, params, **kwargs: (jnp.ones_like(G), None),
    )

    size = int(np.prod(np.asarray(G0.shape)))
    x = jnp.ones((size,), dtype=G0.dtype)
    for key in ("pas-coarse", "hermite-line", "hermite-line-coarse", "identity"):
        _G, _shape, _size, _dt_val, precond_op, _matvec, _squeeze = (
            _build_implicit_operator(
                G0,
                cache,
                params,
                dt=0.2,
                terms=LinearTerms(),
                implicit_preconditioner=key,
            )
        )
        y = precond_op(x)
        assert y.shape == x.shape
        assert np.isfinite(np.asarray(y)).all()

    _G, _shape, _size, _dt_val, callable_precond, _matvec, _squeeze = (
        _build_implicit_operator(
            G0,
            cache,
            params,
            dt=0.2,
            terms=LinearTerms(),
            implicit_preconditioner=lambda x: 2.0 * x,
        )
    )
    np.testing.assert_allclose(np.asarray(callable_precond(x)), np.asarray(2.0 * x))

    with pytest.raises(ValueError):
        _build_implicit_operator(
            G0,
            cache,
            params,
            dt=0.2,
            terms=LinearTerms(),
            implicit_preconditioner="not-a-preconditioner",
        )


def test_build_implicit_operator_linked_hermite_line_preconditioner(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "gkx.solvers_linear_implicit.hypercollision_kz_coefficient", lambda *args: 0.0
    )
    G0 = jnp.zeros((1, 1, 2, 1, 1, 2), dtype=jnp.complex64)
    kz_link = 2.0 * jnp.pi * jnp.fft.fftfreq(2, d=1.0)
    cache = SimpleNamespace(
        lb_lam=jnp.ones((1, 1, 2, 1, 1, 2), dtype=jnp.float32),
        l=jnp.ones((1, 1, 2, 1, 1, 2), dtype=jnp.float32),
        m=jnp.ones((1, 1, 2, 1, 1, 2), dtype=jnp.float32),
        cv_d=jnp.ones((1, 1, 2), dtype=jnp.float32),
        gb_d=jnp.ones((1, 1, 2), dtype=jnp.float32),
        bgrad=jnp.ones((2,), dtype=jnp.float32),
        sqrt_m_ladder=jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        sqrt_p=jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        kz=jnp.asarray([0.0, np.pi], dtype=jnp.float32),
        linked_indices=(jnp.asarray([[0]], dtype=jnp.int32),),
        linked_kz=(kz_link,),
        use_twist_shift=True,
    )
    params = SimpleNamespace(
        nu=jnp.asarray([0.1], dtype=jnp.float32),
        tz=jnp.asarray([1.0], dtype=jnp.float32),
        vth=jnp.asarray([1.0], dtype=jnp.float32),
        omega_d_scale=1.0,
        kpar_scale=1.0,
    )
    monkeypatch.setattr(
        "gkx.solvers_linear_implicit.hypercollision_damping",
        lambda cache, params, dtype: jnp.zeros_like(cache.lb_lam, dtype=dtype),
    )
    monkeypatch.setattr(
        "gkx.solvers_linear_implicit.linear_rhs_cached",
        lambda G, cache, params, **kwargs: (jnp.ones_like(G), None),
    )

    _G, _shape, _size, _dt_val, precond_op, _matvec, _squeeze = (
        _build_implicit_operator(
            G0,
            cache,
            params,
            dt=0.1,
            terms=LinearTerms(),
            implicit_preconditioner="hermite-line",
        )
    )
    y = precond_op(jnp.ones((G0.size,), dtype=G0.dtype))

    assert y.shape == (G0.size,)
    assert np.all(np.isfinite(np.asarray(y)))

    _G, _shape, _size, _dt_val, coarse_precond, _matvec, _squeeze = (
        _build_implicit_operator(
            G0,
            cache,
            params,
            dt=0.1,
            terms=LinearTerms(),
            implicit_preconditioner="hermite-line-coarse",
        )
    )
    y_coarse = coarse_precond(jnp.ones((G0.size,), dtype=G0.dtype))
    assert y_coarse.shape == (G0.size,)
    assert np.all(np.isfinite(np.asarray(y_coarse)))


def test_integrate_linear_wrapper_routes_methods(monkeypatch) -> None:
    G0 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    grid = geom = params = object()
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        "gkx.solvers_linear_integrators.build_linear_cache",
        lambda *args, **kwargs: "cache",
    )
    monkeypatch.setattr(
        "gkx.solvers_linear_integrators._integrate_linear_cached",
        lambda *args, **kwargs: (
            calls.append(("cached", kwargs["method"])) or ("G", "phi")
        ),
    )
    monkeypatch.setattr(
        "gkx.solvers_linear_integrators._integrate_linear_cached_donate",
        lambda *args, **kwargs: (
            calls.append(("donate", kwargs["method"])) or ("Gd", "phid")
        ),
    )
    monkeypatch.setattr(
        "gkx.solvers_linear_integrators._integrate_linear_implicit_cached",
        lambda *args, **kwargs: (
            calls.append(("implicit", "implicit")) or ("Gi", "phii")
        ),
    )

    assert integrate_linear(
        G0, grid, geom, params, dt=0.1, steps=2, method="semi-implicit"
    ) == ("G", "phi")
    assert integrate_linear(
        G0, grid, geom, params, dt=0.1, steps=2, method="rk2", donate=True
    ) == ("Gd", "phid")
    assert integrate_linear(
        G0, grid, geom, params, dt=0.1, steps=2, method="implicit"
    ) == ("Gi", "phii")
    assert ("cached", "imex") in calls
    assert ("donate", "rk2") in calls
    assert ("implicit", "implicit") in calls

    with pytest.raises(ValueError):
        integrate_linear(G0, grid, geom, params, dt=0.1, steps=2, sample_stride=0)
    with pytest.raises(ValueError):
        integrate_linear(G0, grid, geom, params, dt=0.1, steps=3, sample_stride=2)
    with pytest.raises(ValueError):
        integrate_linear(jnp.zeros((2, 2)), grid, geom, params, dt=0.1, steps=2)


def test_cached_linear_binding_resolves_impl_at_trace(monkeypatch) -> None:
    bound = linear_integrators._bind_cached_linear_integrator(donate=False)
    calls: list[bool] = []

    def _fake_impl(G0, *args, **kwargs):
        calls.append(True)
        return G0 + 1.0, G0 + 2.0

    monkeypatch.setattr(linear_integrators, "_integrate_linear_cached_impl", _fake_impl)
    G = jnp.zeros((2,), dtype=jnp.float32)
    state, field = bound(G, G, G, 0.1, 1)

    assert calls == [True]
    np.testing.assert_array_equal(state, jnp.ones_like(G))
    np.testing.assert_array_equal(field, 2.0 * jnp.ones_like(G))


def test_integrate_linear_wrapper_routes_nonserial_parallel(monkeypatch) -> None:
    G0 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    grid = geom = params = object()
    parallel = SimpleNamespace(strategy="velocity", backend="auto")
    calls: list[object] = []

    monkeypatch.setattr(
        "gkx.solvers_linear_integrators.build_linear_cache",
        lambda *args, **kwargs: "cache",
    )
    monkeypatch.setattr(
        "gkx.solvers_linear_integrators._integrate_linear_cached_impl",
        lambda *args, **kwargs: calls.append(kwargs["parallel"]) or ("Gp", "phip"),
    )

    assert integrate_linear(
        G0, grid, geom, params, dt=0.1, steps=2, method="rk2", parallel=parallel
    ) == ("Gp", "phip")
    assert calls == [parallel]

    with pytest.raises(NotImplementedError, match="explicit fixed-step"):
        integrate_linear(
            G0,
            grid,
            geom,
            params,
            dt=0.1,
            steps=2,
            method="implicit",
            parallel=parallel,
        )
    with pytest.raises(NotImplementedError, match="donated"):
        integrate_linear(
            G0,
            grid,
            geom,
            params,
            dt=0.1,
            steps=2,
            method="rk2",
            donate=True,
            parallel=parallel,
        )


def test_integrate_linear_wrapper_enables_electrostatic_field_specialization(
    monkeypatch,
) -> None:
    G0 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    grid = geom = params = object()
    captured: dict[str, bool] = {}

    monkeypatch.setattr(
        "gkx.solvers_linear_integrators.build_linear_cache",
        lambda *args, **kwargs: "cache",
    )

    def _fake_cached(*args, **kwargs):
        captured["force_electrostatic_fields"] = kwargs["force_electrostatic_fields"]
        return "G", "phi"

    monkeypatch.setattr(
        "gkx.solvers_linear_integrators._integrate_linear_cached", _fake_cached
    )

    assert integrate_linear(
        G0,
        grid,
        geom,
        params,
        dt=0.1,
        steps=2,
        method="rk2",
        terms=LinearTerms(apar=0.0, bpar=0.0),
    ) == ("G", "phi")
    assert captured["force_electrostatic_fields"] is True


@pytest.mark.parametrize(
    "terms",
    [
        None,
        LinearTerms(apar=1.0, bpar=0.0),
        LinearTerms(apar=0.0, bpar=1.0),
    ],
)
def test_integrate_linear_wrapper_does_not_force_electrostatic_when_em_terms_enabled(
    monkeypatch,
    terms: LinearTerms | None,
) -> None:
    G0 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    grid = geom = params = object()
    captured: dict[str, bool] = {}

    monkeypatch.setattr(
        "gkx.solvers_linear_integrators.build_linear_cache",
        lambda *args, **kwargs: "cache",
    )

    def _fake_cached(*args, **kwargs):
        captured["force_electrostatic_fields"] = kwargs["force_electrostatic_fields"]
        return "G", "phi"

    monkeypatch.setattr(
        "gkx.solvers_linear_integrators._integrate_linear_cached", _fake_cached
    )

    kwargs = {} if terms is None else {"terms": terms}
    assert integrate_linear(
        G0, grid, geom, params, dt=0.1, steps=2, method="rk2", **kwargs
    ) == ("G", "phi")
    assert captured["force_electrostatic_fields"] is False


def test_linear_rhs_cached_uses_generic_jit_unless_electrostatic_is_forced(
    monkeypatch,
) -> None:
    G0 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    cache = params = object()
    calls: list[str] = []

    def _fake_generic(G, cache, params, terms, dt=None, external_phi=None):
        calls.append("generic")
        assert float(terms.apar) == pytest.approx(0.0)
        assert float(terms.bpar) == pytest.approx(0.0)
        assert float(dt) == pytest.approx(0.25)
        return jnp.zeros_like(G), FieldState(
            phi=jnp.zeros(G.shape[-3:], dtype=G.dtype), apar=None, bpar=None
        )

    monkeypatch.setattr("gkx.terms.assembly.assemble_rhs_cached_jit", _fake_generic)
    monkeypatch.setattr(
        "gkx.terms.assembly.assemble_rhs_cached_electrostatic_jit",
        lambda *args, **kwargs: pytest.fail(
            "electrostatic RHS should run only when explicitly forced"
        ),
    )

    rhs, phi = linear_rhs_cached(
        G0,
        cache,
        params,
        terms=LinearTerms(apar=0.0, bpar=0.0),
        dt=0.25,
    )

    assert calls == ["generic"]
    assert rhs.shape == G0.shape
    assert phi.shape == G0.shape[-3:]


def test_linear_rhs_parallel_cached_serial_alias_and_error_branches(
    monkeypatch,
) -> None:
    G0 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    cache = params = object()
    calls: list[str] = []

    def _fake_serial(G, cache, params, **kwargs):
        calls.append("serial")
        assert kwargs["use_jit"] is False
        assert kwargs["use_custom_vjp"] is False
        assert float(kwargs["dt"]) == pytest.approx(0.2)
        return jnp.ones_like(G), jnp.zeros(G.shape[-3:], dtype=G.dtype)

    monkeypatch.setattr("gkx.operators.linear.rhs.linear_rhs_cached", _fake_serial)

    rhs, phi = linear_rhs_parallel_cached(
        G0,
        cache,
        params,
        terms=LinearTerms(),
        parallel=None,
        use_jit=False,
        use_custom_vjp=False,
        dt=0.2,
    )
    assert calls == ["serial"]
    assert rhs.shape == G0.shape
    assert phi.shape == G0.shape[-3:]

    with pytest.raises(NotImplementedError, match="Hermite axis"):
        linear_rhs_parallel_cached(
            G0,
            cache,
            params,
            terms=LinearTerms(apar=0.0, bpar=0.0),
            parallel=SimpleNamespace(
                strategy="velocity", backend="auto", axis="laguerre"
            ),
        )
    with pytest.raises(NotImplementedError, match="backend='auto'"):
        linear_rhs_parallel_cached(
            G0,
            cache,
            params,
            terms=LinearTerms(),
            parallel=SimpleNamespace(strategy="velocity", backend="auto"),
        )
    with pytest.raises(NotImplementedError, match="currently supports only"):
        linear_rhs_parallel_cached(
            G0,
            cache,
            params,
            terms=LinearTerms(),
            parallel=SimpleNamespace(strategy="kx", backend="auto"),
        )


def test_linear_rhs_parallel_cached_routes_gated_velocity_backends(
    monkeypatch, only_terms
) -> None:
    G0 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    cache = params = object()
    calls: list[tuple[str, int | None]] = []

    streaming_only = only_terms(streaming=1.0)
    electrostatic_slices = only_terms(
        streaming=1.0, mirror=1.0, curvature=1.0, gradb=1.0, diamagnetic=1.0
    )

    def _out(name):
        def _fake(G, cache, params, **kwargs):
            calls.append((name, kwargs.get("num_devices")))
            return jnp.ones_like(G), jnp.zeros(G.shape[-3:], dtype=G.dtype)

        return _fake

    monkeypatch.setattr(
        "gkx.solvers_linear_parallel.linear_rhs_streaming_velocity_sharded",
        _out("streaming"),
    )
    monkeypatch.setattr(
        "gkx.solvers_linear_parallel.linear_rhs_streaming_electrostatic_velocity_sharded",
        _out("streaming_es"),
    )
    monkeypatch.setattr(
        "gkx.solvers_linear_parallel.linear_rhs_electrostatic_slices_velocity_sharded",
        _out("slices"),
    )

    linear_rhs_parallel_cached(
        G0,
        cache,
        params,
        terms=streaming_only,
        parallel=SimpleNamespace(
            strategy="velocity", backend="streaming_only", num_devices=1
        ),
    )
    linear_rhs_parallel_cached(
        G0,
        cache,
        params,
        terms=streaming_only,
        parallel=SimpleNamespace(
            strategy="velocity",
            backend="streaming_electrostatic",
            num_devices=2,
        ),
    )
    linear_rhs_parallel_cached(
        G0,
        cache,
        params,
        terms=electrostatic_slices,
        parallel=SimpleNamespace(
            strategy="velocity",
            backend="electrostatic_linear_slices",
            num_devices=3,
        ),
    )
    linear_rhs_parallel_cached(
        G0,
        cache,
        params,
        terms=electrostatic_slices,
        parallel=SimpleNamespace(strategy="velocity", backend="auto", num_devices=4),
    )

    assert calls == [
        ("streaming", 1),
        ("streaming_es", 2),
        ("slices", 3),
        ("slices", 4),
    ]

    with pytest.raises(NotImplementedError, match="streaming-only"):
        linear_rhs_parallel_cached(
            G0,
            cache,
            params,
            terms=electrostatic_slices,
            parallel=SimpleNamespace(strategy="velocity", backend="streaming_only"),
        )
    with pytest.raises(NotImplementedError, match="collision/EM"):
        linear_rhs_parallel_cached(
            G0,
            cache,
            params,
            terms=LinearTerms(collisions=1.0, apar=0.0, bpar=0.0),
            parallel=SimpleNamespace(
                strategy="velocity", backend="electrostatic_linear_slices"
            ),
        )


def test_linear_rhs_cached_can_use_electrostatic_specialized_jit(monkeypatch) -> None:
    G0 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    cache = params = object()
    calls: list[str] = []

    def _fake_electrostatic(G, cache, params, terms, dt=None, external_phi=None):
        calls.append("electrostatic")
        return jnp.ones_like(G), FieldState(
            phi=jnp.ones(G.shape[-3:], dtype=G.dtype), apar=None, bpar=None
        )

    monkeypatch.setattr(
        "gkx.terms.assembly.assemble_rhs_cached_electrostatic_jit",
        _fake_electrostatic,
    )
    monkeypatch.setattr(
        "gkx.terms.assembly.assemble_rhs_cached_jit",
        lambda *args, **kwargs: pytest.fail(
            "generic RHS should not run when electrostatic specialization is forced"
        ),
    )

    rhs, phi = linear_rhs_cached(
        G0,
        cache,
        params,
        terms=LinearTerms(apar=0.0, bpar=0.0),
        force_electrostatic_fields=True,
    )

    assert calls == ["electrostatic"]
    assert rhs.shape == G0.shape
    assert phi.shape == G0.shape[-3:]


def test_linear_rhs_cached_zero_and_near_zero_states_remain_finite() -> None:
    grid = build_spectral_grid(
        GridConfig(Nx=2, Ny=2, Nz=4, Lx=2.0 * np.pi, Ly=2.0 * np.pi)
    )
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.1)
    params = LinearParams(
        nu=0.0, nu_hyper=0.0, nu_hyper_m=0.0, damp_ends_amp=0.0, damp_ends_widthfrac=0.0
    )
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=2)
    terms = LinearTerms(
        collisions=0.0, hypercollisions=0.0, end_damping=0.0, apar=0.0, bpar=0.0
    )
    G0 = jnp.zeros((2, 2, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)

    dG0, phi0 = linear_rhs_cached(
        G0,
        cache,
        params,
        terms=terms,
        use_jit=False,
        use_custom_vjp=False,
        force_electrostatic_fields=True,
    )
    np.testing.assert_allclose(np.asarray(dG0), 0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(phi0), 0.0, atol=0.0)

    tiny = G0 + jnp.asarray(1.0e-30 + 1.0e-30j, dtype=G0.dtype)
    dG_tiny, phi_tiny = linear_rhs_cached(
        tiny,
        cache,
        params,
        terms=terms,
        use_jit=False,
        use_custom_vjp=False,
        force_electrostatic_fields=True,
    )
    assert np.all(np.isfinite(np.asarray(dG_tiny)))
    assert np.all(np.isfinite(np.asarray(phi_tiny)))


def test_integrate_linear_cached_impl_invalid_and_sampled(monkeypatch) -> None:
    G0 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    cache = SimpleNamespace(lb_lam=jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.float32))
    params = SimpleNamespace(nu=0.0)
    monkeypatch.setattr(
        "gkx.solvers_linear_integrators.hypercollision_damping",
        lambda cache, params, dtype: jnp.zeros_like(cache.lb_lam, dtype=dtype),
    )
    monkeypatch.setattr(
        "gkx.solvers_linear_integrators.linear_rhs_cached",
        lambda G, cache, params, **kwargs: (
            jnp.ones_like(G),
            jnp.zeros((1, 1, 2), dtype=jnp.complex64),
        ),
    )

    with pytest.raises(ValueError):
        _integrate_linear_cached_impl(G0, cache, params, dt=0.1, steps=2, method="bad")

    G_out, phi_t = _integrate_linear_cached_impl(
        G0,
        cache,
        params,
        dt=0.1,
        steps=4,
        method="euler",
        sample_stride=2,
    )
    assert G_out.shape == G0.shape
    assert phi_t.shape[0] == 2


def test_integrate_linear_cached_impl_uses_parallel_rhs(monkeypatch) -> None:
    G0 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    cache = SimpleNamespace(lb_lam=jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.float32))
    params = SimpleNamespace(nu=0.0)
    parallel = SimpleNamespace(strategy="velocity", backend="auto")
    calls: list[object] = []

    monkeypatch.setattr(
        "gkx.solvers_linear_integrators.hypercollision_damping",
        lambda cache, params, dtype: jnp.zeros_like(cache.lb_lam, dtype=dtype),
    )
    monkeypatch.setattr(
        "gkx.solvers_linear_integrators.linear_rhs_cached",
        lambda *args, **kwargs: pytest.fail(
            "serial RHS should not be used for nonserial parallel integration"
        ),
    )

    def _fake_parallel_rhs(G, cache, params, **kwargs):
        calls.append(kwargs["parallel"])
        return jnp.ones_like(G), jnp.zeros((1, 1, 2), dtype=jnp.complex64)

    monkeypatch.setattr(
        "gkx.solvers_linear_integrators.linear_rhs_parallel_cached",
        _fake_parallel_rhs,
    )

    G_out, phi_t = _integrate_linear_cached_impl(
        G0,
        cache,
        params,
        dt=0.1,
        steps=2,
        method="euler",
        parallel=parallel,
    )

    assert G_out.shape == G0.shape
    assert phi_t.shape[0] == 2
    assert calls and all(call is parallel for call in calls)


def test_integrate_linear_implicit_cached_sampled_path(monkeypatch) -> None:
    G0 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    cache = SimpleNamespace()
    params = SimpleNamespace()

    monkeypatch.setattr(
        "gkx.solvers_linear_implicit._build_implicit_operator",
        lambda *args, **kwargs: (
            G0,
            G0.shape,
            G0.size,
            jnp.asarray(0.1, dtype=jnp.float32),
            (lambda x: x),
            (lambda x: x),
            False,
        ),
    )
    monkeypatch.setattr(
        "gkx.solvers_linear_implicit.gmres",
        lambda matvec, rhs, **kwargs: SimpleNamespace(
            x=rhs,
            converged=True,
            residual_norm=jnp.zeros((), dtype=jnp.float32),
            iterations=jnp.asarray(0, dtype=jnp.int32),
        ),
    )
    monkeypatch.setattr(
        "gkx.solvers_linear_implicit.linear_rhs_cached",
        lambda G, cache, params, **kwargs: (
            jnp.ones_like(G),
            jnp.ones((1, 1, 2), dtype=jnp.complex64),
        ),
    )

    G_out, phi_t = _integrate_linear_implicit_cached(
        G0,
        cache,
        params,
        dt=0.1,
        steps=4,
        sample_stride=2,
    )

    assert G_out.shape == G0.shape
    assert phi_t.shape[0] == 2


def test_integrate_linear_diagnostics_validates_and_records_energy(
    diagnostics_cache, patch_diagnostics_kernels
) -> None:
    G0 = jnp.zeros((1, 2, 2, 1, 1, 2), dtype=jnp.complex64)
    grid = geom = params = object()
    cache = diagnostics_cache((1, 2, 2, 1, 1, 2))
    patch_diagnostics_kernels()

    with pytest.raises(ValueError):
        integrate_linear_diagnostics(
            G0, grid, geom, params, dt=0.1, steps=2, sample_stride=0, cache=cache
        )
    with pytest.raises(ValueError):
        integrate_linear_diagnostics(
            G0, grid, geom, params, dt=0.1, steps=3, sample_stride=2, cache=cache
        )

    G_out, phi_t, density_t, hl_t = integrate_linear_diagnostics(
        G0,
        grid,
        geom,
        SimpleNamespace(nu=0.0),
        dt=0.1,
        steps=4,
        method="rk2",
        cache=cache,
        sample_stride=2,
        species_index=0,
        record_hl_energy=True,
    )
    assert G_out.shape == G0.shape
    assert phi_t.shape[0] == 2
    assert density_t.shape[0] == 2
    assert hl_t.shape[0] == 2


@pytest.mark.parametrize("method", ["euler", "rk2", "sspx3", "rk4"])
def test_integrate_linear_diagnostics_explicit_method_branches(
    diagnostics_cache, patch_diagnostics_kernels, method: str
) -> None:
    G0 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    cache = diagnostics_cache((2, 2, 1, 1, 2))
    patch_diagnostics_kernels()

    G_out, phi_t, density_t = integrate_linear_diagnostics(
        G0,
        object(),
        object(),
        SimpleNamespace(nu=0.0),
        dt=0.1,
        steps=2,
        method=method,
        cache=cache,
        sample_stride=1,
        species_index=0,
    )

    assert G_out.shape == G0.shape
    assert phi_t.shape[0] == 2
    assert density_t.shape[0] == 2


def test_integrate_linear_diagnostics_multispecies_density_and_invalid_method(
    diagnostics_cache, patch_diagnostics_kernels
) -> None:
    G0 = jnp.zeros((2, 2, 2, 1, 1, 2), dtype=jnp.complex64)
    cache = diagnostics_cache((2, 2, 2, 1, 1, 2))
    patch_diagnostics_kernels()

    G_out, phi_t, density_t, hl_t = integrate_linear_diagnostics(
        G0,
        object(),
        object(),
        SimpleNamespace(nu=jnp.asarray([0.0, 0.0])),
        dt=0.1,
        steps=2,
        method="rk4",
        cache=cache,
        sample_stride=1,
        species_index=None,
        record_hl_energy=True,
    )

    assert G_out.shape == G0.shape
    assert phi_t.shape[0] == 2
    assert density_t.shape[0] == 2
    assert hl_t.shape[0] == 2
    with pytest.raises(ValueError, match="Unsupported method"):
        integrate_linear_diagnostics(
            G0,
            object(),
            object(),
            SimpleNamespace(nu=jnp.asarray([0.0, 0.0])),
            dt=0.1,
            steps=1,
            method="bad",
            cache=cache,
        )


def test_integrate_linear_diagnostics_builds_cache_and_uses_imex2(
    monkeypatch, diagnostics_cache, patch_diagnostics_kernels
) -> None:
    G0 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    cache = diagnostics_cache((1, 2, 2, 1, 1, 2))
    build_calls: list[tuple[int, int]] = []

    monkeypatch.setattr(
        "gkx.solvers_linear_integrators.build_linear_cache",
        lambda grid, geom, params, Nl, Nm: build_calls.append((Nl, Nm)) or cache,
    )
    patch_diagnostics_kernels()

    G_out, phi_t, density_t = integrate_linear_diagnostics(
        G0,
        object(),
        object(),
        SimpleNamespace(nu=0.0),
        dt=0.1,
        steps=2,
        method="imex2",
        cache=None,
        sample_stride=1,
        species_index=None,
        record_hl_energy=False,
    )

    assert build_calls == [(2, 2)]
    assert G_out.shape == G0.shape
    assert phi_t.shape[0] == 2
    assert density_t.shape[0] == 2


def test_integrate_linear_diagnostics_imex_sampled_multispecies(
    diagnostics_cache, patch_diagnostics_kernels
) -> None:
    G0 = jnp.zeros((2, 2, 2, 1, 1, 2), dtype=jnp.complex64)
    cache = diagnostics_cache((2, 2, 2, 1, 1, 2))
    patch_diagnostics_kernels(rhs=jnp.zeros_like)

    G_out, phi_t, density_t = integrate_linear_diagnostics(
        G0,
        object(),
        object(),
        SimpleNamespace(nu=jnp.asarray([0.1, 0.2], dtype=jnp.float32)),
        dt=0.1,
        steps=4,
        method="imex",
        cache=cache,
        sample_stride=2,
        species_index=None,
        record_hl_energy=False,
    )

    assert G_out.shape == G0.shape
    assert phi_t.shape[0] == 2
    assert density_t.shape == (2, 1, 1, 2)


def test_integrate_linear_diagnostics_species_none_and_5d_density_paths(
    diagnostics_cache, patch_diagnostics_kernels
) -> None:
    cache6 = diagnostics_cache((2, 2, 2, 1, 1, 2))
    cache5 = diagnostics_cache((2, 2, 1, 1, 2))
    patch_diagnostics_kernels()

    G6 = jnp.zeros((2, 2, 2, 1, 1, 2), dtype=jnp.complex64)
    G6_out, phi6_t, density6_t = integrate_linear_diagnostics(
        G6,
        object(),
        object(),
        SimpleNamespace(nu=jnp.asarray([0.0, 0.0])),
        dt=0.1,
        steps=2,
        method="sspx3",
        cache=cache6,
        sample_stride=1,
        species_index=None,
        record_hl_energy=False,
    )
    assert G6_out.shape == G6.shape
    assert phi6_t.shape[0] == 2
    assert density6_t.shape[0] == 2

    G5 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    G5_out, phi5_t, density5_t = integrate_linear_diagnostics(
        G5,
        object(),
        object(),
        SimpleNamespace(nu=0.0),
        dt=0.1,
        steps=2,
        method="rk4",
        cache=cache5,
        sample_stride=1,
        species_index=0,
        record_hl_energy=False,
    )
    assert G5_out.shape == G5.shape
    assert phi5_t.shape[0] == 2
    assert density5_t.shape[0] == 2


@pytest.mark.parametrize(
    ("method", "rate", "damping", "min_order"),
    [
        ("rk2", -0.7 + 0.3j, 0.0, 1.75),
        ("rk4", -0.7 + 0.3j, 0.0, 3.2),
        ("sspx3", -0.7 + 0.3j, 0.0, 2.6),
        ("imex", -0.7 + 0.3j, 0.8, 0.95),
        ("imex2", -0.7 + 0.3j, 0.0, 1.9),
        ("imex2", -0.7 + 0.3j, 0.8, 1.9),
    ],
)
def test_integrate_linear_cached_impl_observed_order_against_exact_solution(
    monkeypatch,
    method: str,
    rate: complex,
    damping: float,
    min_order: float,
) -> None:
    cache = SimpleNamespace(lb_lam=jnp.ones((1, 1, 1, 1, 1), dtype=jnp.float32))
    G0 = jnp.asarray([[[[[1.0 + 0.25j]]]]], dtype=jnp.complex64)
    params = LinearParams(nu=0.0)
    monkeypatch.setattr(
        "gkx.solvers_linear_integrators.hypercollision_damping",
        lambda cache, params, dtype: jnp.ones_like(cache.lb_lam, dtype=dtype) * damping,
    )
    monkeypatch.setattr(
        "gkx.solvers_linear_integrators.linear_rhs_cached",
        lambda G, cache, params, **kwargs: (
            jnp.asarray(rate * G, dtype=G.dtype),
            jnp.asarray(G[0, 0], dtype=G.dtype),
        ),
    )

    t_final = 0.8
    step_sizes: list[float] = []
    errors: list[float] = []
    exact = np.exp(rate * t_final) * np.asarray(G0)
    for steps in (1, 2, 4, 8):
        dt = t_final / steps
        G_out, _phi_t = _integrate_linear_cached_impl(
            G0,
            cache,
            params,
            dt=dt,
            steps=steps,
            method=method,
        )
        errors.append(float(np.max(np.abs(np.asarray(G_out) - exact))))
        step_sizes.append(dt)

    metrics = estimate_observed_order(np.array(step_sizes), np.array(errors))
    assert metrics.asymptotic_order >= min_order


@pytest.mark.parametrize(
    ("collision_weight", "hypercollision_weight", "expected"),
    [(0.0, 0.0, 0.0), (0.25, 0.5, 2.0)],
)
def test_imex_routes_respect_dissipation_term_weights(
    monkeypatch, collision_weight: float, hypercollision_weight: float, expected: float
) -> None:
    modules = (linear_integrators, linear_diagnostics, krylov_algorithms)
    for module in modules:
        monkeypatch.setattr(
            module, "collision_damping", lambda *_args, **_kwargs: jnp.asarray(6.0)
        )
        monkeypatch.setattr(
            module, "hypercollision_damping", lambda *_args: jnp.asarray(1.0)
        )
    terms = LinearTerms(
        collisions=collision_weight, hypercollisions=hypercollision_weight
    )
    state = jnp.asarray([1.0 + 0.0j])
    _, cached = linear_integrators._prepared_linear_state_and_damping(
        state, object(), object(), terms=terms
    )
    diagnostics = linear_diagnostics._linear_damping(
        state, object(), object(), jnp.float32, terms=terms
    )
    krylov = krylov_algorithms._compute_damping(
        state, object(), object(), linear_terms_to_term_config(terms)
    )
    np.testing.assert_allclose([cached, diagnostics, krylov], expected)


def test_imex2_observed_order_for_noncommuting_split() -> None:
    from gkx.solvers_time_explicit_steps import _linear_native_step

    explicit = np.asarray([[0.0, 1.0], [-0.4, 0.2]])
    damping = np.asarray([0.3, 1.1])
    operator = explicit - np.diag(damping)
    eigenvalues, eigenvectors = np.linalg.eig(operator)
    initial = np.asarray([1.0, -0.25])
    exact = eigenvectors @ (
        np.exp(eigenvalues) * np.linalg.solve(eigenvectors, initial)
    )
    step_sizes: list[float] = []
    errors: list[float] = []
    for steps in (4, 8, 16, 32):
        dt = 1.0 / steps
        state = jnp.asarray(initial)
        for _ in range(steps):
            state = _linear_native_step(
                state,
                jnp.asarray(damping),
                jnp.asarray(dt),
                method_key="imex2",
                rhs=lambda value: jnp.asarray(operator) @ value,
            )
        step_sizes.append(dt)
        errors.append(float(np.max(np.abs(np.asarray(state) - exact))))

    metrics = estimate_observed_order(np.asarray(step_sizes), np.asarray(errors))
    assert metrics.asymptotic_order >= 1.9


def test_linked_chain_cover_mask_matches_the_built_cache(spectral_grid) -> None:
    """The light cover builder agrees with the one the eigen routes read off a cache."""

    grid = spectral_grid(
        Nx=8,
        Ny=8,
        Nz=8,
        Lx=2.0 * np.pi,
        Ly=2.0 * np.pi,
        boundary="linked",
        jtwist=1,
    )
    geom = SAlphaGeometry(q=1.4, s_hat=1.0, epsilon=0.1)
    params = LinearParams(nu_hyper=0.0, nu_hyper_m=0.0, damp_ends_widthfrac=0.25)

    cover = linked_chain_cover_mask(grid, geom, params)
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=3)
    from_cache = _linked_covered_mode_mask(cache)

    assert cover is not None and from_cache is not None
    np.testing.assert_array_equal(np.asarray(cover), np.asarray(from_cache))
    # On a full grid the chains reach exactly the two-thirds modes, which is
    # why the nonlinear bracket cannot write outside them.
    np.testing.assert_array_equal(
        np.asarray(cover), np.asarray(grid.dealias_mask, dtype=bool)
    )


def test_mask_off_chain_rows_zeroes_only_the_unreachable_rows(spectral_grid) -> None:
    grid = spectral_grid(
        Nx=8,
        Ny=8,
        Nz=8,
        Lx=2.0 * np.pi,
        Ly=2.0 * np.pi,
        boundary="linked",
        jtwist=1,
    )
    geom = SAlphaGeometry(q=1.4, s_hat=1.0, epsilon=0.1)
    params = LinearParams(nu_hyper=0.0, nu_hyper_m=0.0, damp_ends_widthfrac=0.25)
    cover = np.asarray(linked_chain_cover_mask(grid, geom, params))
    assert not cover.all()

    rng = np.random.default_rng(19)
    shape = (1, 2, 3, int(grid.ky.size), int(grid.kx.size), int(grid.z.size))
    state = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)

    masked = np.asarray(mask_off_chain_rows(state, grid, geom, params))

    off = np.broadcast_to(~cover[:, :, None], shape)
    assert np.max(np.abs(masked[off])) == 0.0
    np.testing.assert_array_equal(masked[~off], state[~off])


def test_mask_off_chain_rows_is_the_identity_on_a_periodic_grid(spectral_grid) -> None:
    grid = spectral_grid(
        Nx=8, Ny=8, Nz=8, Lx=2.0 * np.pi, Ly=2.0 * np.pi, boundary="periodic"
    )
    geom = SAlphaGeometry(q=1.4, s_hat=1.0, epsilon=0.1)
    params = LinearParams(nu_hyper=0.0, nu_hyper_m=0.0, damp_ends_widthfrac=0.25)

    assert linked_chain_cover_mask(grid, geom, params) is None

    rng = np.random.default_rng(20)
    shape = (1, 2, 3, int(grid.ky.size), int(grid.kx.size), int(grid.z.size))
    state = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)

    # Identity, and the same object: a periodic deck adds no array op at all.
    assert mask_off_chain_rows(state, grid, geom, params) is state


# ---- the supplied-state contract's helper (queue row Q23) -----------------
#
# #247 masked the two states the *runtime* does not build. The library entry
# points underneath -- ``gkx.prepare``, the prepared nonlinear ``run``, the
# objective adjoint -- all already hold a built cache, so they read the cover
# off that instead of resolving the twist-shift policy a second time. These
# pin that the two ways of asking agree, and that a periodic deck still gets
# the identity.


def test_mask_supplied_state_matches_the_grid_level_mask(spectral_grid) -> None:
    grid = spectral_grid(
        Nx=8,
        Ny=8,
        Nz=8,
        Lx=2.0 * np.pi,
        Ly=2.0 * np.pi,
        boundary="linked",
        jtwist=1,
    )
    geom = SAlphaGeometry(q=1.4, s_hat=1.0, epsilon=0.1)
    params = LinearParams(nu_hyper=0.0, nu_hyper_m=0.0, damp_ends_widthfrac=0.25)
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=3)
    cover = np.asarray(linked_cover_mask_from_cache(cache))
    assert not cover.all()
    np.testing.assert_array_equal(
        cover, np.asarray(linked_chain_cover_mask(grid, geom, params))
    )

    rng = np.random.default_rng(23)
    shape = (1, 2, 3, int(grid.ky.size), int(grid.kx.size), int(grid.z.size))
    state = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)

    from_cache = np.asarray(mask_supplied_state(state, cache))

    np.testing.assert_array_equal(
        from_cache, np.asarray(mask_off_chain_rows(state, grid, geom, params))
    )
    off = np.broadcast_to(~cover[:, :, None], shape)
    assert np.max(np.abs(from_cache[off])) == 0.0
    np.testing.assert_array_equal(from_cache[~off], state[~off])


def test_mask_supplied_state_is_the_identity_on_a_periodic_cache(spectral_grid) -> None:
    grid = spectral_grid(
        Nx=8, Ny=8, Nz=8, Lx=2.0 * np.pi, Ly=2.0 * np.pi, boundary="periodic"
    )
    geom = SAlphaGeometry(q=1.4, s_hat=1.0, epsilon=0.1)
    params = LinearParams(nu_hyper=0.0, nu_hyper_m=0.0, damp_ends_widthfrac=0.25)
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=3)

    assert linked_cover_mask_from_cache(cache) is None

    rng = np.random.default_rng(24)
    shape = (1, 2, 3, int(grid.ky.size), int(grid.kx.size), int(grid.z.size))
    state = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)

    # The same object, so a periodic deck adds no array op and no graph node.
    assert mask_supplied_state(state, cache) is state


def test_mask_supplied_state_differentiates_without_a_value_branch(
    spectral_grid,
) -> None:
    """The contract is a mask, not a check, so it survives ``grad`` and ``jit``.

    Rejecting a state instead would need its values, which a traced array does
    not have; this is the measurement behind that argument rather than a
    restatement of it.
    """

    grid = spectral_grid(
        Nx=8,
        Ny=8,
        Nz=8,
        Lx=2.0 * np.pi,
        Ly=2.0 * np.pi,
        boundary="linked",
        jtwist=1,
    )
    geom = SAlphaGeometry(q=1.4, s_hat=1.0, epsilon=0.1)
    params = LinearParams(nu_hyper=0.0, nu_hyper_m=0.0, damp_ends_widthfrac=0.25)
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=3)
    cover = np.asarray(linked_cover_mask_from_cache(cache))

    rng = np.random.default_rng(25)
    shape = (1, 2, 3, int(grid.ky.size), int(grid.kx.size), int(grid.z.size))
    state = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)

    masked = jax.jit(lambda s: mask_supplied_state(s, cache))(jnp.asarray(state))
    off = np.broadcast_to(~cover[:, :, None], shape)
    assert np.max(np.abs(np.asarray(masked)[off])) == 0.0

    cotangent = np.asarray(
        jax.grad(lambda s: jnp.sum(jnp.abs(mask_supplied_state(s, cache)) ** 2))(
            jnp.asarray(state)
        )
    )
    assert np.max(np.abs(cotangent[off])) == 0.0
    assert np.max(np.abs(cotangent[~off])) > 0.0


# ---- from test_linear_moments_invariants.py ----
# Fast invariants for Hermite-Laguerre moment primitives.


def _complex_state(shape: tuple[int, ...]) -> jnp.ndarray:
    values = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)
    return (jnp.sin(0.17 * values) + 1j * jnp.cos(0.11 * values)).astype(jnp.complex64)


def _operator_matrix(fn, shape: tuple[int, ...]) -> np.ndarray:
    cols = []
    size = int(np.prod(shape))
    for idx in range(size):
        basis = (
            jnp.zeros(shape, dtype=jnp.float32)
            .reshape(-1)
            .at[idx]
            .set(1.0)
            .reshape(shape)
        )
        cols.append(np.asarray(fn(basis)).reshape(-1))
    return np.stack(cols, axis=1)


def test_linear_moments_reuse_canonical_velocity_space_algebra() -> None:
    """Moment assembly must reuse, rather than copy, velocity recurrences."""

    state = _complex_state((2, 4, 5, 2, 3, 4))
    assert linear_moments.shift_axis is streaming_operators.shift_axis
    assert linear_moments.apply_hermite_v is streaming_operators.apply_hermite_v
    assert linear_moments.apply_hermite_v2 is streaming_operators.apply_hermite_v2
    assert linear_moments.apply_laguerre_x is streaming_operators.apply_laguerre_x

    np.testing.assert_allclose(
        np.asarray(linear_moments.shift_axis(state, 1, axis=-4)),
        np.asarray(streaming_operators.shift_axis(state, 1, axis=-4)),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(linear_moments.shift_axis(state, -2, axis=-5)),
        np.asarray(streaming_operators.shift_axis(state, -2, axis=-5)),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(linear_moments.apply_hermite_v(state)),
        np.asarray(streaming_operators.apply_hermite_v(state)),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(linear_moments.apply_hermite_v2(state)),
        np.asarray(streaming_operators.apply_hermite_v2(state)),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(linear_moments.apply_laguerre_x(state)),
        np.asarray(streaming_operators.apply_laguerre_x(state)),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_velocity_multiplication_matrices_are_self_adjoint() -> None:
    """The truncated Hermite/Laguerre multiplication matrices stay symmetric."""

    shape = (5, 6, 1, 1, 1)
    hermite_v = _operator_matrix(linear_moments.apply_hermite_v, shape)
    hermite_v2 = _operator_matrix(linear_moments.apply_hermite_v2, shape)
    laguerre_x = _operator_matrix(linear_moments.apply_laguerre_x, shape)

    np.testing.assert_allclose(hermite_v, hermite_v.T, rtol=0.0, atol=1.0e-7)
    np.testing.assert_allclose(hermite_v2, hermite_v2.T, rtol=0.0, atol=1.0e-6)
    np.testing.assert_allclose(laguerre_x, laguerre_x.T, rtol=0.0, atol=1.0e-7)
    np.testing.assert_allclose(
        hermite_v2, hermite_v @ hermite_v, rtol=1.0e-6, atol=1.0e-6
    )
    assert np.min(np.linalg.eigvalsh(hermite_v2)) >= -1.0e-6
    assert np.min(np.linalg.eigvalsh(laguerre_x)) > 0.0


def test_periodic_streaming_matches_terms_path_and_conserves_quadratic_norm() -> None:
    """Periodic parallel streaming should be skew-adjoint in the free-energy norm."""

    ns, nl, nm, ny, nx, nz = 2, 3, 5, 2, 1, 16
    z = jnp.linspace(0.0, 2.0 * jnp.pi, nz, endpoint=False)
    dz = z[1] - z[0]
    kz = 2.0 * jnp.pi * jnp.fft.fftfreq(nz, d=dz)
    state = _complex_state((ns, nl, nm, ny, nx, nz))
    state = state * (1.0 + 0.1 * jnp.sin(z)).reshape((1, 1, 1, 1, 1, nz))
    vth = jnp.asarray([0.7, 1.3], dtype=jnp.float32)

    linear_path = linear_moments.streaming_term(state, dz=dz, vth=vth)
    sqrt_p, sqrt_m = hermite_ladder_coeffs(nm - 1)
    sqrt_shape = (1, 1, nm, 1, 1, 1)
    modular = streaming_operators.streaming_ladder_term(
        state,
        kz=kz,
        vth=vth.reshape((ns, 1, 1, 1, 1, 1)),
        sqrt_p=sqrt_p[:nm].reshape(sqrt_shape),
        sqrt_m=sqrt_m[:nm].reshape(sqrt_shape),
    )

    np.testing.assert_allclose(
        np.asarray(linear_path), np.asarray(modular), rtol=1.0e-6, atol=1.0e-6
    )
    energy_rate = jnp.real(jnp.vdot(state, linear_path))
    assert abs(float(energy_rate)) < 2.0e-4


# ---- from test_linear_param_units.py ----
#


CYCLONE_TOML = REPO_ROOT / "benchmarks/cases/cyclone_nonlinear_t400.toml"

# Cyclone base case, as the literature quotes it.
LITERATURE_R_OVER_LT = 6.9
LITERATURE_R_OVER_LN = 2.2


@pytest.fixture(scope="module")
def cyclone_toml() -> dict:
    return tomllib.loads(CYCLONE_TOML.read_text(encoding="utf-8"))


def test_the_runtime_hands_the_operator_the_toml_tprim_unscaled(cyclone_toml):
    from gkx.workflows.runtime.startup import build_runtime_linear_params
    from gkx.workflows.runtime.toml import load_runtime_from_toml

    cfg, _raw = load_runtime_from_toml(CYCLONE_TOML)
    params = build_runtime_linear_params(cfg)

    ion = cyclone_toml["species"][0]
    assert np.asarray(params.tprim).ravel()[0] == pytest.approx(float(ion["tprim"]))
    assert np.asarray(params.fprim).ravel()[0] == pytest.approx(float(ion["fprim"]))


def test_only_R_over_a_turns_the_stored_drive_into_the_literature_gradient(
    cyclone_toml,
):
    # The cyclone normalization contract sets a = 1, so R/a is the TOML's own R0.
    assert cyclone_toml["normalization"]["contract"] == "cyclone"
    r_over_a = float(cyclone_toml["geometry"]["R0"])
    ion = cyclone_toml["species"][0]

    assert float(ion["tprim"]) * r_over_a == pytest.approx(
        LITERATURE_R_OVER_LT, abs=0.02
    )
    assert float(ion["fprim"]) * r_over_a == pytest.approx(
        LITERATURE_R_OVER_LN, abs=0.03
    )
    # Stated, not folded in: the stored value is not the literature one.
    assert float(ion["tprim"]) != pytest.approx(LITERATURE_R_OVER_LT, abs=0.5)


def test_both_gradient_defaults_are_the_shipped_case_in_the_units_the_operator_reads(
    cyclone_toml,
):
    from gkx.config import ModelConfig
    from gkx.operators.linear.params import LinearParams

    ion = cyclone_toml["species"][0]
    assert LinearParams().tprim == pytest.approx(float(ion["tprim"]))
    assert LinearParams().fprim == pytest.approx(float(ion["fprim"]))
    assert ModelConfig().tprim_i == pytest.approx(float(ion["tprim"]))
    assert ModelConfig().fprim == pytest.approx(float(ion["fprim"]))


def test_no_public_gradient_name_claims_a_normalization_that_is_never_applied():
    from gkx import api
    from gkx.config import KineticElectronModelConfig, ModelConfig
    from gkx.operators.linear.params import LinearParams, Species

    classes = (LinearParams, Species, ModelConfig, KineticElectronModelConfig)
    named = {field.name for cls in classes for field in dataclasses.fields(cls)} | set(
        api.__all__
    )
    offenders = sorted(name for name in named if name.startswith("R_over_L"))
    assert offenders == []


def test_the_retired_names_still_work_and_say_why_they_are_wrong():
    from gkx.operators.linear.params import (
        DEPRECATED_LINEAR_PARAM_ALIASES,
        LinearParams,
    )

    assert DEPRECATED_LINEAR_PARAM_ALIASES == {
        "R_over_LTi": "tprim",
        "R_over_Ln": "fprim",
        "R_over_LTe": "tprim_e",
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        params = LinearParams(R_over_LTi=6.9, R_over_Ln=2.2, R_over_LTe=1.0)
    assert params.tprim == 6.9
    assert params.fprim == 2.2
    assert params.tprim_e == 1.0
    assert len(caught) == 3
    assert all(issubclass(w.category, DeprecationWarning) for w in caught)
    # The warning has to name the unit, not just the replacement: a caller who
    # passed 6.9 meant R/L_T and is now getting a/L_T.
    assert "a/L" in str(caught[0].message)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert params.R_over_LTi == 6.9
    assert len(caught) == 1

    # dataclasses.replace routes through __init__, so the alias survives it.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        replaced = dataclasses.replace(params, R_over_Ln=3.0)
    assert replaced.fprim == 3.0
    assert replaced.tprim == 6.9
    assert len(caught) == 1


# ---- from test_dimits_threshold_extraction.py ----
# Numerics: does the linear-threshold estimator recover a planted root?
#
# The critical gradient sets the window for the nonlinear Dimits scan, so an
# estimator that returns a confident wrong number costs GPU time on an
# uninterpretable run. Three earlier versions of ``scripts/campaigns/dimits_shift.py``
# reduced over ``k_y`` before fitting, which mixes branches by construction: the


def _mode(x: float, critical: float, slope: float) -> dict[str, float]:
    """One point on ``gamma = slope (x - x_crit)``, damped below the root."""

    return {"gamma": float(slope * (x - critical)), "omega": -0.25}


def _rows(multipliers, modes: dict[str, tuple[float, float]]) -> list[dict]:
    return [
        {
            "multiplier": float(x),
            "by_ky": {key: _mode(x, *params) for key, params in modes.items()},
        }
        for x in multipliers
    ]


def test_minimum_over_ky_reports_the_earliest_mode_not_the_strongest():
    module = load_tool_script("campaigns", "dimits_shift")
    # ky#3 grows faster wherever both are unstable, but ky#5 goes unstable first,
    # and the case's critical gradient belongs to whichever mode is first.
    rows = _rows(np.linspace(0.20, 0.80, 31), {"3": (0.55, 0.40), "5": (0.35, 0.10)})

    result = module.threshold_from_scan(rows)

    assert result["resolved"]
    assert result["critical_ky_index"] == 5
    assert result["critical_multiplier"] == pytest.approx(0.35, abs=0.01)


def test_a_root_outside_its_sign_change_bracket_is_refused():
    module = load_tool_script("campaigns", "dimits_shift")
    rows = _rows(np.linspace(0.20, 0.80, 31), {"4": (0.45, 0.25)})
    # One stray unstable point far below the root, which is what a continuation
    # that jumped to a neighbouring branch produces. The curve now changes sign
    # at 0.30 while the fit window still straddles 0.45, so the two disagree --
    # and that disagreement, not the fit residual, is what has to stop the
    # number.
    for row in rows:
        if row["multiplier"] == pytest.approx(0.30):
            row["by_ky"]["4"] = {"gamma": 0.09, "omega": -0.25}

    result = module.threshold_from_scan(rows)

    assert not result["resolved"]
    assert result["per_ky"]["4"]["in_bracket"] is False
    assert result["per_ky"]["4"]["sign_change_bracket"] == pytest.approx([0.28, 0.30])


def test_ntft_linked_derivative_follows_the_ballooning_mode() -> None:
    """On a non-twisting flux tube row ``kx`` at ``z`` holds the ballooning
    mode ``kx + m0(ky, z)``, so the parallel derivative runs along
    ``kx + m0 = const``. A smooth bump carried by one ballooning mode must
    differentiate to the bump's derivative on that path; holding the row
    index fixed instead cuts the bump wherever ``m0`` jumps (20% error)."""

    grid = build_spectral_grid(
        GridConfig(
            Nx=16,
            Ny=8,
            Nz=64,
            Lx=2.0 * np.pi,
            Ly=2.0 * np.pi,
            y0=10.0,
            boundary="linked",
            jtwist=1,
            non_twist=True,
            ky_layout="full",
        )
    )
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.1)
    cache = build_linear_cache(
        grid, geom, LinearParams(nu_hyper=0.0, nu_hyper_m=0.0), Nl=1, Nm=1
    )
    ky, z = 2, np.asarray(grid.z)
    rows = np.mod(-np.asarray(cache.ntft_m0)[ky], grid.kx.size)
    assert np.ptp(rows) > 0
    bump = np.exp(-((z / 0.8) ** 2))
    G = np.zeros((grid.ky.size, grid.kx.size, z.size), dtype=complex)
    G[ky, rows, np.arange(z.size)] = bump
    dG = grad_z_linked_fft(
        jnp.asarray(G),
        dz=cache.dz,
        linked_indices=cache.linked_indices,
        linked_kz=cache.linked_kz,
        linked_inverse_permutation=cache.linked_inverse_permutation,
        linked_full_cover=cache.linked_full_cover,
        linked_gather_map=cache.linked_gather_map,
        linked_gather_mask=cache.linked_gather_mask,
        linked_use_gather=cache.linked_use_gather,
        ntft_m0=cache.ntft_m0,
    )
    np.testing.assert_allclose(
        np.asarray(dG)[ky, rows, np.arange(z.size)],
        -2.0 * z / 0.64 * bump,
        atol=1.0e-4,
    )
