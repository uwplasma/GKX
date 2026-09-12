"""Tests for modular field solves and custom-VJP behavior."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gkx.config import CycloneBaseCase, GridConfig
from gkx.geometry import SAlphaGeometry
from gkx.core_grid import build_spectral_grid
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.moments import quasineutrality_phi
from gkx.operators.linear.params import LinearParams
from gkx.parallel.velocity_drive import electrostatic_phi_reference
from gkx.terms.fields import _solve_fields_impl, solve_fields


@pytest.mark.parametrize("dtype,tol", [(jnp.float32, 3e-6), (jnp.float64, 3e-13)])
@pytest.mark.parametrize("beta", [1e-6, 0.02, 0.2])
@pytest.mark.parametrize("ell,m", [(0, 0), (0, 1), (1, 0)])
@pytest.mark.parametrize("variable_b", [False, True])
def test_three_field_dense_system_independent_moments(
    dtype, tol, beta, ell, m, variable_b
):
    """GX Eqs. 32--34 at B=1; variable B tests GKX's implementation convention."""
    from gkx.core_velocity import J_l_all

    cache, params, *_ = _build_case(beta=beta, fapar=1.0)
    b = np.array([[0.2, 0.5, 0.8], [0.04, 0.1, 0.16]])
    # Analytic Laguerre coefficients, not a field/cache helper's reduction.
    jl = np.stack(
        [np.exp(-b / 2), -b / 2 * np.exp(-b / 2), b * b / 8 * np.exp(-b / 2)], axis=1
    )
    np.testing.assert_allclose(
        np.moveaxis(J_l_all(jnp.asarray(b, dtype), 2), 0, 1), jl, rtol=tol
    )
    jb = jl + np.concatenate([np.zeros_like(jl[:, :1]), jl[:, :-1]], axis=1)
    B = np.array([0.8, 1.0, 1.3]) if variable_b else np.ones(3)
    k2 = np.array([0.3, 0.4, 0.7])
    cache = replace(
        cache,
        Jl=jnp.asarray(jl[:, :, None, None, :], dtype),
        JlB=jnp.asarray(jb[:, :, None, None, :], dtype),
        bmag=jnp.asarray(B, dtype),
        kperp2=jnp.asarray(k2[None, None, :], dtype),
        jacobian=jnp.ones(3, dtype),
        ky=jnp.array([0.2], dtype),
        mask0=jnp.zeros((1, 1, 3), dtype=bool),
        kperp2_bmag=False,
    )
    params = replace(
        params, tau_e=0.0, apar_beta_scale=0.5, ampere_g0_scale=0.5, bpar_beta_scale=0.5
    )
    z, n, T, mass = (
        np.array([1.0, -1.0]),
        np.ones(2),
        np.array([1.0, 1.7]),
        np.array([1.0, 0.01]),
    )
    vth = np.sqrt(T / mass)
    G = np.zeros((2, 3, 2, 1, 1, 3), dtype=np.complex128)
    G[:, ell, m, 0, 0, :] = np.array([1 + 0.3j, -0.4 + 0.8j])[:, None] * np.array(
        [1.0, 0.7, 1.2]
    )
    out = solve_fields(
        jnp.asarray(G, jnp.complex64 if dtype == jnp.float32 else jnp.complex128),
        cache,
        params,
        **{
            key: jnp.asarray(value, dtype)
            for key, value in dict(
                charge=z,
                density=n,
                temp=T,
                mass=mass,
                tz=T / z,
                vth=vth,
                fapar=1.0,
                w_bpar=1.0,
            ).items()
        },
    )
    # The paper's Eq. 34 has no B^-2. At variable B this independently
    # assembles GKX's current convention, whose physical normalization is open.
    for iz in range(3):
        M, rhs = np.diag([0.0, k2[iz], 1.0]), np.zeros(3, dtype=complex)
        for s in range(2):
            j, p = jl[s, :, iz], jb[s, :, iz]
            g0, g1 = G[s, :, 0, 0, 0, iz], G[s, :, 1, 0, 0, iz]
            M[0, 0] += n[s] * z[s] ** 2 / T[s] * (1 - j @ j)
            M[0, 2] -= n[s] * z[s] * (j @ p)
            M[2, 0] += beta / 2 / B[iz] ** 2 * n[s] * z[s] * (j @ p)
            M[2, 2] += beta / 2 / B[iz] ** 2 * n[s] * T[s] * (p @ p)
            M[1, 1] += beta / 2 * n[s] * z[s] ** 2 / mass[s] * (j @ j)
            rhs += [
                n[s] * z[s] * (j @ g0),
                beta / 2 * n[s] * z[s] * vth[s] * (j @ g1),
                -beta / 2 / B[iz] ** 2 * n[s] * T[s] * (p @ g0),
            ]
        actual = np.array([out.phi[0, 0, iz], out.apar[0, 0, iz], out.bpar[0, 0, iz]])
        expected = np.linalg.solve(M, rhs)
        np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol * 1e-3)
        scale = np.linalg.norm(M) * np.linalg.norm(actual) + np.linalg.norm(rhs)
        assert np.linalg.norm(M @ actual - rhs) / scale < tol
        # A sign mutation must not satisfy this independently assembled system.
        assert np.linalg.norm(M @ (-actual) - rhs) / scale > 100 * tol


def _build_case(
    *,
    beta: float,
    fapar: float,
) -> tuple[
    object,
    LinearParams,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    grid_cfg = GridConfig(Nx=4, Ny=4, Nz=8, Lx=62.8, Ly=62.8)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(
        beta=beta,
        fapar=fapar,
        tau_e=1.0,
        nu=0.0,
        nu_hyper=0.0,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
    )
    cache = build_linear_cache(grid, geom, params, Nl=3, Nm=4)
    ny, nx, nz = grid.ky.size, grid.kx.size, grid.z.size
    G = jnp.zeros((1, 3, 4, ny, nx, nz), dtype=jnp.complex64)
    z = jnp.asarray(grid.z, dtype=jnp.float32)
    phase = jnp.exp(1j * z)
    G = G.at[0, 0, 0, 1, 1, :].set(0.2 * phase)
    G = G.at[0, 1, 0, 1, 1, :].set(0.1 * phase)
    G = G.at[0, 0, 1, 1, 1, :].set(0.05j * phase)
    charge = jnp.asarray([1.0], dtype=jnp.float32)
    density = jnp.asarray([1.0], dtype=jnp.float32)
    temp = jnp.asarray([1.0], dtype=jnp.float32)
    mass = jnp.asarray([1.0], dtype=jnp.float32)
    tz = jnp.asarray([1.0], dtype=jnp.float32)
    vth = jnp.asarray([1.0], dtype=jnp.float32)
    return cache, params, G, charge, density, temp, mass, tz, vth


def test_solve_fields_matches_impl_and_mask0_zeroing() -> None:
    cache, params, G, charge, density, temp, mass, tz, vth = _build_case(
        beta=0.05, fapar=1.0
    )
    fapar = jnp.asarray(1.0, dtype=jnp.float32)
    w_bpar = jnp.asarray(1.0, dtype=jnp.float32)
    out_impl = _solve_fields_impl(
        G,
        cache,
        params,
        charge=charge,
        density=density,
        temp=temp,
        mass=mass,
        tz=tz,
        vth=vth,
        fapar=fapar,
        w_bpar=w_bpar,
    )
    out_vjp = solve_fields(
        G,
        cache,
        params,
        charge=charge,
        density=density,
        temp=temp,
        mass=mass,
        tz=tz,
        vth=vth,
        fapar=fapar,
        w_bpar=w_bpar,
    )
    assert jnp.allclose(out_vjp.phi, out_impl.phi, rtol=1.0e-6, atol=1.0e-6)
    assert jnp.allclose(out_vjp.apar, out_impl.apar, rtol=1.0e-6, atol=1.0e-6)
    assert jnp.allclose(out_vjp.bpar, out_impl.bpar, rtol=1.0e-6, atol=1.0e-6)

    mask0 = jnp.broadcast_to(cache.mask0, out_impl.phi.shape)
    assert jnp.allclose(out_impl.phi[mask0], 0.0)
    assert jnp.allclose(out_impl.apar[mask0], 0.0)
    assert jnp.allclose(out_impl.bpar[mask0], 0.0)


def test_solve_fields_bpar_and_apar_toggles() -> None:
    cache, params, G, charge, density, temp, mass, tz, vth = _build_case(
        beta=0.05, fapar=1.0
    )
    out_off = _solve_fields_impl(
        G,
        cache,
        params,
        charge=charge,
        density=density,
        temp=temp,
        mass=mass,
        tz=tz,
        vth=vth,
        fapar=jnp.asarray(0.0, dtype=jnp.float32),
        w_bpar=jnp.asarray(0.0, dtype=jnp.float32),
    )
    assert jnp.allclose(out_off.apar, 0.0)
    assert jnp.allclose(out_off.bpar, 0.0)

    phi_es = quasineutrality_phi(G, cache.Jl, params.tau_e, charge, density, tz)
    phi_es = jnp.where(cache.mask0, 0.0, phi_es)
    assert jnp.allclose(out_off.phi, phi_es, rtol=1.0e-6, atol=1.0e-6)

    params_beta0 = replace(params, beta=0.0)
    out_beta0 = _solve_fields_impl(
        G,
        cache,
        params_beta0,
        charge=charge,
        density=density,
        temp=temp,
        mass=mass,
        tz=tz,
        vth=vth,
        fapar=jnp.asarray(1.0, dtype=jnp.float32),
        w_bpar=jnp.asarray(1.0, dtype=jnp.float32),
    )
    assert jnp.allclose(out_beta0.bpar, 0.0)


def test_electrostatic_field_solve_allows_single_hermite_moment() -> None:
    grid_cfg = GridConfig(Nx=2, Ny=4, Nz=8, Lx=12.0, Ly=12.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(beta=0.0, fapar=0.0, tau_e=1.0)
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=1)
    G = jnp.zeros(
        (1, 2, 1, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64
    )
    G = G.at[0, 0, 0, 1, 0, :].set(0.1 + 0.2j)

    out = _solve_fields_impl(
        G,
        cache,
        params,
        charge=jnp.asarray([1.0], dtype=jnp.float32),
        density=jnp.asarray([1.0], dtype=jnp.float32),
        temp=jnp.asarray([1.0], dtype=jnp.float32),
        mass=jnp.asarray([1.0], dtype=jnp.float32),
        tz=jnp.asarray([1.0], dtype=jnp.float32),
        vth=jnp.asarray([1.0], dtype=jnp.float32),
        fapar=jnp.asarray(0.0, dtype=jnp.float32),
        w_bpar=jnp.asarray(0.0, dtype=jnp.float32),
    )

    assert out.phi.shape == (grid.ky.size, grid.kx.size, grid.z.size)
    assert jnp.allclose(out.apar, 0.0)
    assert jnp.allclose(out.bpar, 0.0)
    assert jnp.all(jnp.isfinite(out.phi))


def test_solve_fields_custom_vjp_gradient_matches_impl() -> None:
    cache, params, G, charge, density, temp, mass, tz, vth = _build_case(
        beta=0.05, fapar=1.0
    )
    fapar = jnp.asarray(1.0, dtype=jnp.float32)
    w_bpar = jnp.asarray(1.0, dtype=jnp.float32)

    def loss_impl(G_in: jnp.ndarray) -> jnp.ndarray:
        out = _solve_fields_impl(
            G_in,
            cache,
            params,
            charge=charge,
            density=density,
            temp=temp,
            mass=mass,
            tz=tz,
            vth=vth,
            fapar=fapar,
            w_bpar=w_bpar,
        )
        return jnp.real(jnp.sum(jnp.abs(out.phi) ** 2))

    def loss_vjp(G_in: jnp.ndarray) -> jnp.ndarray:
        out = solve_fields(
            G_in,
            cache,
            params,
            charge=charge,
            density=density,
            temp=temp,
            mass=mass,
            tz=tz,
            vth=vth,
            fapar=fapar,
            w_bpar=w_bpar,
        )
        return jnp.real(jnp.sum(jnp.abs(out.phi) ** 2))

    g_impl = jax.grad(loss_impl)(G)
    g_vjp = jax.grad(loss_vjp)(G)
    assert jnp.all(jnp.isfinite(g_impl))
    assert jnp.all(jnp.isfinite(g_vjp))
    assert jnp.allclose(g_vjp, g_impl, rtol=1.0e-5, atol=1.0e-5)


def test_adiabatic_zonal_field_solve_uses_cached_jacobian() -> None:
    cache, params, G, charge, density, temp, mass, tz, vth = _build_case(
        beta=0.0, fapar=0.0
    )
    G = jnp.zeros_like(G)
    z = jnp.asarray(cache.bmag)
    zonal_profile = 0.2 + 0.05j * z
    G = G.at[0, 0, 0, 0, 1, :].set(zonal_profile)

    out_base = _solve_fields_impl(
        G,
        cache,
        params,
        charge=charge,
        density=density,
        temp=temp,
        mass=mass,
        tz=tz,
        vth=vth,
        fapar=jnp.asarray(0.0, dtype=jnp.float32),
        w_bpar=jnp.asarray(0.0, dtype=jnp.float32),
    )
    varied_jacobian = jnp.linspace(
        1.0, 3.0, cache.jacobian.size, dtype=cache.jacobian.dtype
    )
    out_varied = _solve_fields_impl(
        G,
        replace(cache, jacobian=varied_jacobian),
        params,
        charge=charge,
        density=density,
        temp=temp,
        mass=mass,
        tz=tz,
        vth=vth,
        fapar=jnp.asarray(0.0, dtype=jnp.float32),
        w_bpar=jnp.asarray(0.0, dtype=jnp.float32),
    )

    assert not jnp.allclose(out_base.phi, out_varied.phi)


def test_serial_reference_matches_canonical_zonal_value_and_gradient() -> None:
    cache, params, G, charge, density, temp, mass, tz, vth = _build_case(
        beta=0.0, fapar=0.0
    )
    G = (
        jnp.zeros_like(G)
        .at[0, 0, 0, 0, 1, :]
        .set(0.2 + 0.05j * jnp.asarray(cache.bmag))
    )

    def production_phi(G_in: jnp.ndarray) -> jnp.ndarray:
        return _solve_fields_impl(
            G_in,
            cache,
            params,
            charge=charge,
            density=density,
            temp=temp,
            mass=mass,
            tz=tz,
            vth=vth,
            fapar=jnp.asarray(0.0, dtype=jnp.float32),
            w_bpar=jnp.asarray(0.0, dtype=jnp.float32),
        ).phi

    def reference_phi(G_in: jnp.ndarray) -> jnp.ndarray:
        return electrostatic_phi_reference(
            G_in,
            Jl=cache.Jl,
            tau_e=params.tau_e,
            charge=charge,
            density=density,
            tz=tz,
            mask0=cache.mask0,
            jacobian=cache.jacobian,
            ky=cache.ky,
        )

    expected = production_phi(G)
    observed = jax.jit(reference_phi)(G)
    assert jnp.allclose(observed, expected, rtol=1.0e-6, atol=1.0e-6)

    def loss(phi_fn, state):
        return jnp.real(jnp.sum(jnp.abs(phi_fn(state)) ** 2))

    expected_grad = jax.grad(lambda state: loss(production_phi, state))(G)
    observed_grad = jax.grad(lambda state: loss(reference_phi, state))(G)
    assert jnp.allclose(observed_grad, expected_grad, rtol=1.0e-5, atol=1.0e-5)
