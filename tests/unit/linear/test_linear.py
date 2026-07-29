"""Linear operator tests for the flux-tube electrostatic model."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.integration

from gkx.config import CycloneBaseCase, GridConfig, GeometryConfig
from gkx.diagnostics.analysis import estimate_observed_order
from gkx.geometry import SAlphaGeometry, SlabGeometry, sample_flux_tube_geometry
from gkx.core.grid import build_spectral_grid, select_ky_grid
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
from gkx.solvers.linear.integrators import integrate_linear
from gkx.operators.linear.linked import _build_linked_fft_maps
from gkx.operators.linear.params import _x64_enabled
from gkx.operators.linear.streaming import grad_z_linked_fft
from gkx.core.velocity import J_l_all
from gkx.solvers.linear.krylov import dominant_eigenpair
from gkx.solvers.linear.implicit import _build_implicit_operator
from gkx.terms.linear_terms import (
    collision_invariant_rates,
    collision_quadratic_rate,
    collisions_contribution,
    conservative_full_f_dougherty_cross_moments,
    drift_kinetic_coulomb_six_moment_contribution,
    drift_kinetic_dougherty_contribution,
    drift_kinetic_sugama_six_moment_contribution,
    multispecies_collision_invariant_rates,
)
from gkx.terms.assembly import assemble_rhs_terms_cached
from gkx.terms.config import TermConfig


def test_grad_z_periodic_sine():
    """Centered periodic derivative should differentiate a sine wave."""
    z = jnp.linspace(0.0, 2.0 * jnp.pi, 64, endpoint=False)
    dz = z[1] - z[0]
    f = jnp.sin(z)
    df = grad_z_periodic(f, dz)
    assert jnp.allclose(df, jnp.cos(z), atol=2.0e-2)


def test_build_linked_fft_maps_keeps_real_fft_positive_ky_modes():
    kx = np.array([0.0], dtype=float)
    ky = np.array([0.0, 0.01, 0.02], dtype=float)
    linked_indices, linked_kz = _build_linked_fft_maps(
        kx=kx,
        ky=ky,
        y0=100.0,
        jtwist=2,
        dz=(2.0 * np.pi) / 32.0,
        nz=32,
        real_dtype=jnp.float32,
        ky_mode=np.array([0, 1, 2], dtype=int),
    )

    assert len(linked_indices) == 1
    assert np.array_equal(
        np.asarray(linked_indices[0]), np.array([[0], [1], [2]], dtype=np.int32)
    )
    assert np.asarray(linked_kz[0]).shape == (32,)


def test_build_linear_cache_zero_shat_periodic_uses_periodic_fft_without_end_damping():
    from gkx.geometry import SlabGeometry, apply_geometry_grid_defaults
    from gkx.config import GeometryConfig
    from gkx.core.grid import select_real_fft_ky_grid
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


def test_compute_b_shape_and_value():
    """b should match k_perp^2 for s-alpha geometry."""
    grid_cfg = GridConfig(Nx=8, Ny=6, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    b = compute_b(grid, geom, rho=1.0)
    assert b.shape == (cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz)
    kx0 = grid.kx[0]
    ky0 = grid.ky[0]
    theta0 = grid.z[0]
    kx_eff = kx0 + geom.s_hat * ky0 * theta0
    assert jnp.isclose(b[0, 0, 0], kx_eff * kx_eff + ky0 * ky0)


def test_slab_itg_matrix_matches_published_gyro_moment_hierarchy() -> None:
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
        R_over_Ln=jnp.asarray([1.0]),
        R_over_LTi=jnp.asarray([3.0]),
    )
    cache = build_linear_cache(
        grid, SlabGeometry(s_hat=0.0, z0=10.0), params, Nl=nl, Nm=nm
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
    mode_count = nl * nm
    observed = np.empty((mode_count, mode_count), dtype=complex)
    for column in range(mode_count):
        laguerre_order, hermite_order = divmod(column, nm)
        state = np.zeros((1, nl, nm, 4, 1, 8), dtype=complex)
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
    grid_cfg = GridConfig(Nx=1, Ny=1, Nz=8, Lx=6.0, Ly=6.0)
    grid = build_spectral_grid(grid_cfg)
    geom_base = SAlphaGeometry.from_config(GeometryConfig(R0=2.77778))
    geom_bmag = SAlphaGeometry.from_config(
        GeometryConfig(R0=2.77778, kperp2_bmag=False, bessel_bmag_power=1.0)
    )
    params = LinearParams()
    cache_base = build_linear_cache(grid, geom_base, params, Nl=2, Nm=2)
    cache_bmag = build_linear_cache(grid, geom_bmag, params, Nl=2, Nm=2)
    bmag = geom_bmag.bmag(jnp.asarray(grid.z))
    ratio = cache_bmag.b / cache_base.b
    expected = (1.0 / bmag)[None, None, None, :]
    mask = jnp.isfinite(ratio)
    assert jnp.allclose(ratio[mask], expected[mask], rtol=1.0e-6, atol=1.0e-8)


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


def test_build_linear_cache_restores_linked_end_damping_on_full_fft_grid():
    grid = build_spectral_grid(
        GridConfig(
            Nx=8,
            Ny=8,
            Nz=8,
            Lx=62.8,
            Ly=2.0 * np.pi,
            boundary="linked",
            y0=1.0,
        )
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


def test_build_linear_cache_keeps_linked_end_damping_on_selected_positive_ky_grid():
    grid_full = build_spectral_grid(
        GridConfig(
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


def test_assemble_rhs_terms_scales_linked_end_damping_by_step_dt():
    grid_full = build_spectral_grid(
        GridConfig(
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
    )
    ky_idx = int(np.argmin(np.abs(np.asarray(grid_full.ky) - 0.3)))
    grid = select_ky_grid(grid_full, ky_idx)
    geom = SAlphaGeometry.from_config(GeometryConfig(s_hat=0.8))
    params = LinearParams(damp_ends_amp=0.1, damp_ends_widthfrac=0.125)
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=4)
    G = jnp.ones((2, 4, 1, 1, 96), dtype=jnp.complex64)
    term_cfg = TermConfig(
        streaming=0.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=1.0,
        apar=0.0,
        bpar=0.0,
    )

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
    assert np.allclose(end_dt[mask], end_raw[mask] / 0.2, rtol=1.0e-6, atol=1.0e-6)


def test_streaming_zero_for_constant_z():
    """Streaming should vanish for z-constant fields."""
    grid_cfg = GridConfig(Nx=8, Ny=6, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
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

    G = jnp.zeros((2, 3, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    G = G.at[:, 1:, ...].set(1.0)
    terms = LinearTerms(
        streaming=1.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )
    dG, _phi = linear_rhs(G, grid, geom, params, terms=terms)
    assert jnp.allclose(dG, 0.0)


def test_linear_rhs_shapes():
    """RHS and potential should have consistent shapes."""
    grid_cfg = GridConfig(Nx=8, Ny=6, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 3, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    dG, phi = linear_rhs(G, grid, geom, params)
    assert dG.shape == G.shape
    assert phi.shape == (cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz)


def test_linear_param_validation():
    """Invalid parameters should be rejected in checked paths."""
    grid_cfg = GridConfig(Nx=8, Ny=6, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    G = jnp.zeros((2, 3, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    with pytest.raises(ValueError):
        compute_b(grid, geom, rho=0.0)
    with pytest.raises(ValueError):
        quasineutrality_phi(
            G[None, ...],
            jnp.ones((1, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz)),
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


def test_integrate_linear_shapes():
    """Integrator should return a time series of phi with expected length."""
    grid_cfg = GridConfig(Nx=8, Ny=6, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    _, phi_t = integrate_linear(G, grid, geom, params, dt=0.1, steps=3, method="rk4")
    assert phi_t.shape[0] == 3


def test_integrate_linear_progress_with_sample_stride_gt_one():
    """Sampled progress reporting must compute diagnostics before emitting callbacks."""
    grid_cfg = GridConfig(Nx=4, Ny=4, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))

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


def test_integrate_linear_methods():
    """Explicit and IMEX paths should run without error."""
    grid_cfg = GridConfig(Nx=6, Ny=6, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    for method in ("euler", "rk2", "imex", "semi-implicit", "sspx3"):
        _, phi_t = integrate_linear(
            G, grid, geom, params, dt=0.1, steps=2, method=method
        )
        assert phi_t.shape[0] == 2


def test_integrate_linear_with_cache():
    """Integrate with a precomputed cache path."""
    grid_cfg = GridConfig(Nx=6, Ny=6, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    cache = build_linear_cache(grid, geom, params, G.shape[0], G.shape[1])
    _, phi_t = integrate_linear(
        G, grid, geom, params, dt=0.1, steps=2, method="rk4", cache=cache
    )
    assert phi_t.shape[0] == 2


def test_integrate_linear_checkpoint_runs():
    """Checkpointed integration should run on a tiny grid."""
    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    _, phi_t = integrate_linear(
        G, grid, geom, params, dt=0.1, steps=2, method="rk4", checkpoint=True
    )
    assert phi_t.shape[0] == 2


def test_integrate_linear_invalid_method():
    """Invalid integrator names should raise a ValueError."""
    grid_cfg = GridConfig(Nx=6, Ny=6, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    with pytest.raises(ValueError):
        integrate_linear(G, grid, geom, params, dt=0.1, steps=2, method="rk5")


def test_linear_cache_matches_rhs():
    """Cached RHS should match the direct RHS for the same inputs."""
    grid_cfg = GridConfig(Nx=6, Ny=6, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    cache = build_linear_cache(grid, geom, params, G.shape[0], G.shape[1])
    dG0, phi0 = linear_rhs(G, grid, geom, params)
    dG1, phi1 = linear_rhs_cached(G, cache, params)
    assert jnp.allclose(dG0, dG1)
    assert jnp.allclose(phi0, phi1)


def test_linear_cached_rhs_replaces_builtin_collision_operator():
    """A custom collision model replaces collisions but not other terms."""

    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.ones((1, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex64)
    cache = build_linear_cache(grid, geom, params, Nl=1, Nm=2)
    terms = LinearTerms(
        streaming=0.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.25,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )

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


def test_linear_cached_rhs_rejects_invalid_collision_shape():
    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.ones((1, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex64)
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


def test_linear_integrator_applies_custom_collision_each_step():
    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G0 = jnp.ones((1, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex64)

    class DragCollision:
        def apply(self, context):
            return -3.0 * context.distribution

    terms = LinearTerms(
        streaming=0.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.25,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )
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


def test_linear_cache_tree_roundtrip():
    """LinearCache pytree should round-trip through flatten/unflatten."""
    grid_cfg = GridConfig(Nx=4, Ny=4, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=2)
    children, aux = cache.tree_flatten()
    cache2 = LinearCache.tree_unflatten(aux, children)
    assert jnp.allclose(cache2.Jl, cache.Jl)
    assert cache2.kperp2_bmag is cache.kperp2_bmag
    assert jnp.allclose(cache2.omega_d, cache.omega_d)
    assert jnp.allclose(cache2.lb_lam, cache.lb_lam)
    assert jnp.allclose(cache2.hyper_ratio, cache.hyper_ratio)


def test_build_linear_cache_multispecies():
    """Cache should support multiple species arrays."""
    grid_cfg = GridConfig(Nx=4, Ny=4, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(rho=jnp.array([1.0, 0.5]))
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=2)
    assert cache.Jl.shape[0] == 2


def test_linear_rhs_multispecies_shapes():
    """Multispecies RHS should return a matching shape."""
    grid_cfg = GridConfig(Nx=4, Ny=4, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(
        charge_sign=jnp.array([1.0, -1.0]),
        density=jnp.array([1.0, 1.0]),
        mass=jnp.array([1.0, 0.001]),
        temp=jnp.array([1.0, 1.0]),
        vth=jnp.array([1.0, 1.0]),
        rho=jnp.array([1.0, 0.5]),
        tz=jnp.array([1.0, -1.0]),
        R_over_Ln=jnp.array([0.0, 0.0]),
        R_over_LTi=jnp.array([0.0, 0.0]),
    )
    G = jnp.zeros((2, 2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex64)
    cache = build_linear_cache(grid, geom, params, Nl=G.shape[1], Nm=G.shape[2])
    dG, phi = linear_rhs_cached(G, cache, params)
    assert dG.shape == G.shape
    assert phi.shape == (cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz)


def test_implicit_preconditioner_hermite_line_shape_and_finite():
    """Hermite-line preconditioner should run and preserve shape/dtype."""

    grid_cfg = GridConfig(Nx=2, Ny=4, Nz=16, Lx=62.8, Ly=62.8)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(
        R_over_Ln=cfg.model.R_over_Ln,
        R_over_LTi=cfg.model.R_over_LTi,
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


def test_implicit_preconditioner_linked_hermite_line_coarse_shape_and_finite():
    """Linked Hermite-line coarse preconditioner should preserve finite vectors."""

    grid_cfg = GridConfig(Nx=4, Ny=4, Nz=8, Lx=6.28, Ly=6.28, boundary="linked")
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
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


def _streaming_shift_invert_setup():
    grid_cfg = GridConfig(Nx=2, Ny=4, Nz=16, Lx=6.28, Ly=6.28)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
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
    terms = LinearTerms(
        streaming=1.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )
    return v0, cache, params, terms


def test_shift_invert_nearest_pair_passes_physical_outer_residual():
    """Nearest-shift selection should admit a converged streaming Ritz pair."""

    v0, cache, params, terms = _streaming_shift_invert_setup()
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


def test_shift_invert_preconditioner_rejects_unconverged_outer_pair():
    """A completed inner solve must not promote a large-residual Ritz pair."""

    v0, cache, params, terms = _streaming_shift_invert_setup()
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


def test_linear_cache_rho_star_scales_ky():
    """rho_star should scale cached ky for normalization control."""
    grid_cfg = GridConfig(Nx=4, Ny=4, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(rho_star=2.0)
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=2)
    assert jnp.allclose(cache.ky, grid.ky * 2.0)


def test_linear_rhs_cached_invalid_shape():
    """Cached RHS should reject invalid shapes."""
    grid_cfg = GridConfig(Nx=4, Ny=4, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=2)
    with pytest.raises(ValueError):
        linear_rhs_cached(jnp.zeros((2, 3, 4)), cache, params)


def test_jit_path_handles_tracers():
    """JIT tracing should exercise the tracer-safe validation path."""
    import jax

    grid_cfg = GridConfig(Nx=8, Ny=6, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))

    @jax.jit
    def _run(G_in):
        return linear_rhs(G_in, grid, geom, params)[0]

    out = _run(G)
    assert out.shape == G.shape


def test_integrate_linear_implicit_runs():
    """Implicit path should run on a tiny grid."""
    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(nu=0.1)
    G = jnp.zeros((1, 1, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
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


def test_mirror_curvature_terms_activate_with_drift_scale():
    """Drift/mirror terms should activate when omega_d_scale is nonzero."""
    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    G = jnp.zeros((1, 3, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex64)
    G = G.at[0, 1, 1, 0, :].set(1.0 + 0.0j)

    params_off = LinearParams(
        omega_d_scale=0.0,
        omega_star_scale=0.0,
        kpar_scale=0.0,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
    )
    cache_off = build_linear_cache(grid, geom, params_off, G.shape[0], G.shape[1])
    terms_off = LinearTerms(
        streaming=0.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
    )
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
    terms_on = LinearTerms(
        streaming=0.0,
        mirror=1.0,
        curvature=1.0,
        gradb=1.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
    )
    dG_on, _phi_on = linear_rhs_cached(G, cache_on, params_on, terms=terms_on)
    assert jnp.max(jnp.abs(dG_on)) > 0.0


def test_diamagnetic_drive_populates_second_hermite_moment():
    """Diamagnetic drive should populate the m=2 component when enabled."""
    grid_cfg = GridConfig(Nx=2, Ny=4, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    G = jnp.zeros((2, 3, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex64)
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
    terms_off = LinearTerms(
        streaming=0.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
    )
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
    terms_on = LinearTerms(
        streaming=0.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=1.0,
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
    )
    dG_on, _phi_on = linear_rhs_cached(G, cache_on, params_on, terms=terms_on)
    assert jnp.max(jnp.abs(dG_on[:, 2, ...])) > 0.0
    assert jnp.allclose(dG_on[:, 1, ...], 0.0)


def test_diamagnetic_drive_vanishes_for_zonal_mode():
    """Diamagnetic drive should vanish for the ky=0 mode."""
    grid_cfg = GridConfig(Nx=2, Ny=4, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    G = jnp.zeros((2, 3, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex64)
    G = G.at[0, 0, 0, 0, :].set(1.0 + 0.0j)
    params = LinearParams(
        omega_d_scale=0.0,
        omega_star_scale=1.0,
        kpar_scale=0.0,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
    )
    cache = build_linear_cache(grid, geom, params, G.shape[0], G.shape[1])
    terms = LinearTerms(
        streaming=0.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=1.0,
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
    )
    dG, _phi = linear_rhs_cached(G, cache, params, terms=terms)
    assert jnp.allclose(dG, 0.0)


def test_rho_star_scales_cache_ky():
    """rho_star should scale the cached ky values."""
    grid_cfg = GridConfig(Nx=4, Ny=4, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(rho_star=2.0)
    cache = build_linear_cache(grid, geom, params, Nl=1, Nm=1)
    assert jnp.allclose(cache.ky, 2.0 * grid.ky)


def test_shift_axis_noop():
    """shift_axis should return the input when offset is zero."""
    from gkx.operators.linear.moments import shift_axis

    arr = jnp.arange(6.0).reshape(2, 3)
    out = shift_axis(arr, 0, axis=0)
    assert jnp.allclose(out, arr)


def test_apar_streaming_coupling_changes_rhs():
    """Finite beta should modify streaming via Apar coupling."""
    grid_cfg = GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)

    G = jnp.zeros((2, 3, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex64)
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
