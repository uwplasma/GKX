"""Tests for modular field solves and custom-VJP behavior."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import eval_laguerre, j1

from gkx.config import CycloneBaseCase, GeometryConfig, GridConfig
from gkx.core_ky_layout import transport_mode_weights
from gkx.geometry import SAlphaGeometry
from gkx.core_grid import build_spectral_grid
from gkx.operators.fluxes import particle_flux_channel_species
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.moments import build_H, quasineutrality_phi
from gkx.operators.linear.params import LinearParams
from gkx.operators.moments import fieldline_quadrature_weights
from gkx.parallel.velocity_drive import electrostatic_phi_reference
from gkx.terms.assembly import assemble_rhs_terms_cached
from gkx.terms.config import TermConfig
from gkx.terms.fields import _solve_fields_impl, solve_fields
from gkx.terms.linear_terms import linked_streaming_contribution, mirror_contribution


def _analytic_gyro_coefficients(b):
    jl = np.stack(
        [np.exp(-b / 2), -b / 2 * np.exp(-b / 2), b**2 / 8 * np.exp(-b / 2)], axis=1
    )
    return jl, jl + np.concatenate([np.zeros_like(jl[:, :1]), jl[:, :-1]], axis=1)


def _direct_bpar_laguerre_coefficients(b, nl):
    """Project 2 mu B J1(alpha)/alpha by independent Gauss-Laguerre quadrature."""

    nodes, weights = np.polynomial.laguerre.laggauss(96)
    alpha = np.sqrt(2.0 * nodes[:, None] * np.ravel(b)[None, :])
    kernel = 2.0 * nodes[:, None] * j1(alpha) / alpha
    coeff = [
        np.sum(
            weights[:, None]
            * (-1) ** ell
            * eval_laguerre(ell, nodes)[:, None]
            * kernel,
            axis=0,
        )
        for ell in range(nl)
    ]
    return np.stack(coeff).reshape((nl,) + np.shape(b))


def _electromagnetic_species(dtype):
    values = {
        "charge": np.array([1.0, -1.0], dtype=dtype),
        "density": np.array([0.9, 1.1], dtype=dtype),
        "temp": np.array([1.0, 1.7], dtype=dtype),
        "mass": np.array([1.3, 0.04], dtype=dtype),
    }
    values["vth"] = np.sqrt(values["temp"] / values["mass"])
    values["tz"] = values["temp"] / values["charge"]
    rho = np.sqrt(values["temp"] * values["mass"]) / np.abs(values["charge"])
    jax_values = {key: jnp.asarray(value, dtype) for key, value in values.items()}
    nt = jnp.asarray(values["density"] * values["temp"], dtype)
    return values, jax_values, rho, nt


def _three_field_source_quadratic(G, cache, params, source_args):
    jl, jb, zweight, values, jax_values, iy, ix = source_args
    fields = solve_fields(G, cache, params, fapar=1.0, w_bpar=1.0, **jax_values)
    g0, g1 = G[:, :, 0, iy, ix], G[:, :, 1, iy, ix]
    density, charge = values["density"], values["charge"]
    nt = jax_values["density"] * jax_values["temp"]
    moments = jnp.stack(
        [
            jnp.sum((density * charge)[:, None, None] * jl * g0, axis=(0, 1)),
            -jnp.sum(
                (density * charge * values["vth"])[:, None, None] * jl * g1,
                axis=(0, 1),
            ),
            jnp.sum(nt[:, None, None] * jb * g0, axis=(0, 1)),
        ]
    )
    kinetic = 0.5 * jnp.sum(
        zweight[None, None, None, :]
        * nt[:, None, None, None]
        * jnp.abs(G[:, :, :, iy, ix]) ** 2
    )
    field_values = jnp.stack(
        [fields.phi[iy, ix], fields.apar[iy, ix], fields.bpar[iy, ix]]
    )
    return kinetic + 0.5 * jnp.real(
        jnp.sum(zweight[None, :] * jnp.conj(field_values) * moments)
    )


def _assert_three_field_residual(
    out, G, jl, jb, B, k2, beta, charge, density, temp, mass, vth, tol, iy=0, ix=0
):
    for iz in range(B.size):
        M, rhs = np.diag([0.0, k2[iz], 1.0]), np.zeros(3, dtype=complex)
        for s in range(charge.size):
            j, p = jl[s, :, iz], jb[s, :, iz]
            g0, g1 = G[s, :, 0, iy, ix, iz], G[s, :, 1, iy, ix, iz]
            M[0, 0] += density[s] * charge[s] ** 2 / temp[s] * (1 - j @ j)
            M[0, 2] -= density[s] * charge[s] * (j @ p)
            M[2, 0] += beta / 2 / B[iz] ** 2 * density[s] * charge[s] * (j @ p)
            M[2, 2] += beta / 2 / B[iz] ** 2 * density[s] * temp[s] * (p @ p)
            M[1, 1] += beta / 2 * density[s] * charge[s] ** 2 / mass[s] * (j @ j)
            rhs += [
                density[s] * charge[s] * (j @ g0),
                beta / 2 * density[s] * charge[s] * vth[s] * (j @ g1),
                -beta / 2 / B[iz] ** 2 * density[s] * temp[s] * (p @ g0),
            ]
        actual = np.array(
            [out.phi[iy, ix, iz], out.apar[iy, ix, iz], out.bpar[iy, ix, iz]]
        )
        expected = np.linalg.solve(M, rhs)
        np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol * 1e-3)
        scale = np.linalg.norm(M) * np.linalg.norm(actual) + np.linalg.norm(rhs)
        assert np.linalg.norm(M @ actual - rhs) / scale < tol
        assert np.linalg.norm(M @ (-actual) - rhs) / scale > 100 * tol


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
    jl, jb = _analytic_gyro_coefficients(b)
    np.testing.assert_allclose(
        np.moveaxis(J_l_all(jnp.asarray(b, dtype), 2), 0, 1), jl, rtol=tol
    )
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
    _assert_three_field_residual(out, G, jl, jb, B, k2, beta, z, n, T, mass, vth, tol)


@pytest.mark.parametrize("kperp2_bmag", [True, False])
def test_geometry_flr_matches_independent_three_field_residual(kperp2_bmag):
    """Validate declared GKX FLR/B conventions, not physical B normalization."""
    dtype = np.float64 if jax.config.x64_enabled else np.float32
    tol = 3e-13 if jax.config.x64_enabled else 5e-6
    grid = build_spectral_grid(
        GridConfig(Nx=1, Ny=4, Nz=8, Lx=8.0, Ly=7.0, boundary="periodic")
    )
    geom = SAlphaGeometry.from_config(
        GeometryConfig(
            R0=2.77778,
            epsilon=0.18,
            kperp2_bmag=kperp2_bmag,
            bessel_bmag_power=1.0,
        )
    )
    values, jax_values, rho, nt = _electromagnetic_species(dtype)
    charge, density, temp, mass, vth = (
        values[key] for key in ("charge", "density", "temp", "mass", "vth")
    )
    beta = 0.04
    params = LinearParams(
        beta=beta, fapar=1.0, tau_e=0.0, rho=jnp.asarray(rho), rho_star=0.8
    )
    cache = build_linear_cache(grid, geom, params, Nl=3, Nm=2)
    iy, ix = 1, 0
    G = np.zeros((2, 3, 2, grid.ky.size, grid.kx.size, grid.z.size), complex)
    phase = np.array([1.0, 0.7, 1.2, 0.8, 1.1, 0.6, 0.9, 1.3])
    G[:, :, 0, iy, ix] = np.array([[0.3 + 0.2j], [-0.4 + 0.5j]])[:, :, None] * phase
    G[:, :, 1, iy, ix] = np.array([[0.1 - 0.3j], [0.2 + 0.4j]])[:, :, None] * phase
    ctype = jnp.complex128 if dtype == np.float64 else jnp.complex64
    out = solve_fields(
        jnp.asarray(G, ctype),
        cache,
        params,
        fapar=1.0,
        w_bpar=1.0,
        **jax_values,
    )
    theta = jnp.asarray(grid.z, dtype=dtype)
    gds2, gds21, gds22 = (np.asarray(x) for x in geom.metric_coeffs(theta))
    ky = dtype(params.rho_star * grid.ky[iy])
    kx_hat = dtype(params.rho_star * grid.kx[ix]) / dtype(geom.s_hat)
    k2 = ky * (ky * gds2 + 2 * kx_hat * gds21) + kx_hat**2 * gds22
    B = np.asarray(geom.bmag(theta))
    b = rho[:, None] ** 2 * k2[None, :] / B[None, :] ** (2 * kperp2_bmag + 1)
    jl, jb = _analytic_gyro_coefficients(b)
    assert 0 < iy < grid.ky.size - 1 and b.size > 0 and np.all(b > 0) and np.ptp(B) > 0
    _assert_three_field_residual(
        out, G, jl, jb, B, k2, beta, charge, density, temp, mass, vth, tol, iy, ix
    )

    # GX v3 Eqs. 12, 16, and 32--34 imply this discrete source quadratic.
    # At constant B it checks GKX's G -> H algebra, not a physical energy norm.
    flat_geom = replace(geom, epsilon=0.0)
    flat_cache = build_linear_cache(grid, flat_geom, params, Nl=3, Nm=2)
    flat_out = solve_fields(
        jnp.asarray(G, ctype), flat_cache, params, fapar=1.0, w_bpar=1.0, **jax_values
    )
    flat_B = np.asarray(flat_geom.bmag(theta))
    flat_b = rho[:, None] ** 2 * k2[None, :] / flat_B[None, :] ** (2 * kperp2_bmag + 1)
    flat_jl, flat_jb = _analytic_gyro_coefficients(flat_b)
    assert np.ptp(flat_B) == 0.0 and all(
        np.any(np.abs(np.asarray(field[iy, ix])) > 0)
        for field in (flat_out.phi, flat_out.apar, flat_out.bpar)
    )
    jl_jax, jb_jax = jnp.asarray(flat_jl, dtype), jnp.asarray(flat_jb, dtype)
    G_jax = jnp.asarray(G, ctype)
    gradient = jax.grad(_three_field_source_quadratic)(
        G_jax,
        flat_cache,
        params,
        (
            jl_jax,
            jb_jax,
            jnp.ones(grid.z.size, dtype=dtype),
            values,
            jax_values,
            iy,
            ix,
        ),
    )
    H = build_H(
        G_jax,
        flat_cache.Jl,
        flat_out.phi,
        jax_values["tz"],
        flat_out.apar,
        jax_values["vth"],
        flat_out.bpar,
        flat_cache.JlB,
    )
    np.testing.assert_allclose(
        np.asarray(jnp.conj(gradient[:, :, :, iy, ix])),
        np.asarray(nt[:, None, None, None] * H[:, :, :, iy, ix]),
        rtol=tol,
        atol=tol,
    )
    streamed = linked_streaming_contribution(
        G_jax,
        phi=flat_out.phi,
        apar=flat_out.apar,
        bpar=flat_out.bpar,
        Jl=flat_cache.Jl,
        JlB=flat_cache.JlB,
        tz=jax_values["tz"],
        vth=jax_values["vth"],
        sqrt_p=flat_cache.sqrt_p,
        sqrt_m=flat_cache.sqrt_m_ladder,
        kpar_scale=jnp.asarray(params.kpar_scale, dtype),
        weight=jnp.asarray(1.0, dtype),
        kz=flat_cache.kz,
        dz=flat_cache.dz,
        hermite_closure="truncation",
    )
    # Truncation keeps the Hermite ladder symmetric; periodic d/dz is skew-adjoint.
    # This isolates streaming and makes no curved-geometry or nonlinear claim.
    exchange = (
        nt[:, None, None, None]
        * jnp.conj(H[:, :, :, iy, ix])
        * streamed[:, :, :, iy, ix]
    )
    scale = jnp.sum(jnp.abs(exchange))
    assert scale > 0.0
    assert jnp.abs(jnp.real(jnp.sum(exchange))) < tol * scale


def test_variable_b_conservative_linear_weighted_exchange_converges():
    """Periodic streaming/mirror exchange converges in the volume measure.

    This is a term-level gate, not a full free-energy budget with drives,
    collisions, damping, curvature, or nonlinear transfer.
    """

    dtype = np.float64 if jax.config.x64_enabled else np.float32
    ctype = jnp.complex128 if dtype == np.float64 else jnp.complex64
    tol = 3e-12 if dtype == np.float64 else 2e-5
    values, jax_values, rho, nt = _electromagnetic_species(dtype)
    params = LinearParams(
        beta=0.04,
        fapar=1.0,
        tau_e=0.0,
        charge_sign=jax_values["charge"],
        density=jax_values["density"],
        temp=jax_values["temp"],
        mass=jax_values["mass"],
        tz=jax_values["tz"],
        vth=jax_values["vth"],
        rho=jnp.asarray(rho, dtype),
        rho_star=0.8,
        kpar_scale=1.0 / (1.4 * 2.77778),
        fprim=jnp.zeros(2, dtype),
        tprim=jnp.zeros(2, dtype),
    )
    geom = SAlphaGeometry(
        q=1.4,
        s_hat=0.0,
        epsilon=0.18,
        R0=2.77778,
        kperp2_bmag=True,
        bessel_bmag_power=1.0,
    )

    defects = []
    wrong_weights = []
    wrong_signs = []
    isolated_rates = []
    total_defects = []
    drift_defects = {"curvature": [], "gradb": []}
    drift_mutations = []
    conservative_terms = TermConfig(
        collisions=0.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
    )
    for nz in (16, 32, 64, 128):
        grid = build_spectral_grid(
            GridConfig(Nx=1, Ny=4, Nz=nz, Lx=8.0, Ly=7.0, boundary="periodic")
        )
        cache = build_linear_cache(grid, geom, params, Nl=3, Nm=4)
        iy, ix = 1, 0
        z = np.asarray(grid.z, dtype=dtype)
        G = np.zeros((2, 3, 4, grid.ky.size, grid.kx.size, nz), complex)
        s, ell, m = np.indices((2, 3, 4))
        k1, k2 = 1 + (s + ell + 2 * m) % 5, 1 + (2 * s + 2 * ell + m) % 6
        amp = 0.02 * (1 + s + ell) + 0.015j * (1 + m)
        G[:, :, :, iy, ix] = amp[..., None] * (
            np.exp(1j * k1[..., None] * z)
            + 0.31 * np.exp(-1j * k2[..., None] * z)
            + 0.17 * np.cos(7.0 * z)
        )
        G_jax = jnp.asarray(G, ctype)
        total_rhs, fields, contributions = assemble_rhs_terms_cached(
            G_jax,
            cache,
            params,
            terms=conservative_terms,
            use_custom_vjp=False,
        )
        assert all(
            np.linalg.norm(np.asarray(field[iy, ix])) > 0.0
            for field in (fields.phi, fields.apar, fields.bpar)
        )

        theta = jnp.asarray(grid.z, dtype=dtype)
        B = np.asarray(geom.bmag(theta))
        k2 = np.asarray(cache.kperp2[iy, ix]) * B**2
        b = rho[:, None] ** 2 * k2[None, :] / B[None, :] ** 3
        jl, jb = _analytic_gyro_coefficients(b)
        jl_jax, jb_jax = jnp.asarray(jl, dtype), jnp.asarray(jb, dtype)
        independent_jacobian = 1.0 / (abs(geom.gradpar()) * B)
        independent_vol = jnp.asarray(
            independent_jacobian / independent_jacobian.sum(), dtype
        )
        vol, _ = fieldline_quadrature_weights(geom, grid)
        np.testing.assert_allclose(
            np.asarray(vol), np.asarray(independent_vol), rtol=tol
        )

        H = build_H(
            G_jax,
            cache.Jl,
            fields.phi,
            jax_values["tz"],
            fields.apar,
            jax_values["vth"],
            fields.bpar,
            cache.JlB,
        )
        gradient = jax.grad(_three_field_source_quadratic)(
            G_jax,
            cache,
            params,
            (jl_jax, jb_jax, independent_vol, values, jax_values, iy, ix),
        )
        expected_gradient = (
            independent_vol[None, None, None, :]
            * nt[:, None, None, None]
            * H[:, :, :, iy, ix]
        )
        np.testing.assert_allclose(
            np.asarray(jnp.conj(gradient[:, :, :, iy, ix])),
            np.asarray(expected_gradient),
            rtol=tol,
            atol=tol,
        )

        streaming = linked_streaming_contribution(
            G_jax,
            phi=fields.phi,
            apar=fields.apar,
            bpar=fields.bpar,
            Jl=cache.Jl,
            JlB=cache.JlB,
            tz=jax_values["tz"],
            vth=jax_values["vth"],
            sqrt_p=cache.sqrt_p,
            sqrt_m=cache.sqrt_m_ladder,
            kpar_scale=jnp.asarray(params.kpar_scale, dtype),
            weight=jnp.asarray(1.0, dtype),
            kz=cache.kz,
            dz=cache.dz,
            hermite_closure="truncation",
        )
        mirror = mirror_contribution(
            H,
            vth=jax_values["vth"],
            bgrad=cache.bgrad,
            ell=cache.l,
            sqrt_m=cache.sqrt_m,
            sqrt_m_p1=cache.sqrt_m_p1,
            weight=jnp.asarray(1.0, dtype),
        )

        def normalized_rate(rhs, zweight):
            exchange = (
                zweight[None, None, None, :]
                * nt[:, None, None, None]
                * jnp.conj(H[:, :, :, iy, ix])
                * rhs[:, :, :, iy, ix]
            )
            scale = jnp.sum(jnp.abs(exchange))
            assert scale > 0.0
            return float(jnp.abs(jnp.real(jnp.sum(exchange))) / scale)

        defects.append(normalized_rate(streaming + mirror, independent_vol))
        wrong_weights.append(
            normalized_rate(streaming + mirror, jnp.ones(nz, dtype=dtype) / nz)
        )
        wrong_signs.append(normalized_rate(streaming - mirror, independent_vol))
        isolated_rates.append(
            (
                normalized_rate(streaming, independent_vol),
                normalized_rate(mirror, independent_vol),
            )
        )
        total_defects.append(normalized_rate(total_rhs, independent_vol))
        for name in ("curvature", "gradb"):
            contribution = contributions[name]
            assert np.linalg.norm(np.asarray(contribution)) > 10 * tol
            drift_defects[name].append(normalized_rate(contribution, independent_vol))
            drift_mutations.append(normalized_rate(1j * contribution, independent_vol))
        for name in (
            "diamagnetic",
            "collisions",
            "hypercollisions",
            "hyperdiffusion",
            "end_damping",
        ):
            np.testing.assert_allclose(np.asarray(contributions[name]), 0.0, atol=tol)

    min_reduction = 100 if dtype == np.float64 else 32
    resolved_tol = tol if dtype == np.float64 else 8 * np.finfo(dtype).eps
    assert defects[0] > min_reduction * max(defects[1:])
    assert max(defects[1:]) < resolved_tol
    assert total_defects[0] > min_reduction * max(total_defects[1:])
    assert max(total_defects[1:]) < resolved_tol
    assert (
        max(rate for rates in drift_defects.values() for rate in rates) < resolved_tol
    )
    assert min(rate for pair in isolated_rates for rate in pair) > 10 * tol
    assert min(drift_mutations) > 10 * tol
    assert min(wrong_weights[-2:]) > 10 * tol
    assert min(wrong_signs[-2:]) > 10 * tol


def test_variable_b_bpar_local_normalization_pressure_and_flux():
    """Check local-beta balance and Bpar factors in H and particle flux."""

    dtype = np.float64 if jax.config.x64_enabled else np.float32
    ctype = jnp.complex128 if dtype == np.float64 else jnp.complex64
    tol = 2e-11 if dtype == np.float64 else 3e-5
    grid = build_spectral_grid(GridConfig(Nx=1, Ny=4, Nz=12, Lx=8.0, Ly=7.0))
    geom = SAlphaGeometry(
        q=1.4,
        s_hat=0.0,
        epsilon=0.22,
        R0=2.77778,
        kperp2_bmag=True,
        bessel_bmag_power=0.0,
    )
    beta = 0.06
    species = {
        "charge": jnp.asarray([1.0, -1.0], dtype),
        "density": jnp.ones(2, dtype),
        "temp": jnp.ones(2, dtype),
        "mass": jnp.ones(2, dtype),
        "tz": jnp.asarray([1.0, -1.0], dtype),
        "vth": jnp.ones(2, dtype),
    }
    params = LinearParams(
        beta=beta,
        tau_e=0.0,
        rho=jnp.asarray([0.7, 0.7], dtype),
        density=species["density"],
        temp=species["temp"],
        tz=species["tz"],
        vth=species["vth"],
    )
    cache = build_linear_cache(grid, geom, params, Nl=3, Nm=1)
    iy, ix = 1, 0
    z = np.asarray(grid.z)
    G = np.zeros((2, 3, 1, grid.ky.size, grid.kx.size, grid.z.size), complex)
    pressure_seed = 0.18 + 0.04 * np.cos(z) + 0.03j * np.sin(2.0 * z)
    G[:, 0, 0, iy, ix] = pressure_seed
    out = solve_fields(
        jnp.asarray(G, ctype), cache, params, fapar=0.0, w_bpar=1.0, **species
    )

    B = np.asarray(cache.bmag)
    direct_jb = np.moveaxis(
        _direct_bpar_laguerre_coefficients(np.asarray(cache.b[:, iy, ix]), 3), 0, 1
    )
    cached_jb = np.asarray(cache.JlB[:, :, iy, ix])
    np.testing.assert_allclose(cached_jb, direct_jb, rtol=tol, atol=tol)
    nt = np.ones(2)[:, None, None]
    pressure = np.sum(nt * direct_jb * G[:, :, 0, iy, ix], axis=(0, 1))
    susceptibility = np.sum(nt * direct_jb**2, axis=(0, 1))
    local_beta_half = beta / (2.0 * B**2)
    expected_bpar = (
        -local_beta_half * pressure / (1.0 + local_beta_half * susceptibility)
    )
    actual_bpar = np.asarray(out.bpar[iy, ix])
    np.testing.assert_allclose(np.asarray(out.phi[iy, ix]), 0.0, atol=tol)
    np.testing.assert_allclose(actual_bpar, expected_bpar, rtol=tol, atol=tol)
    total_pressure = pressure + susceptibility * actual_bpar
    balance = B**2 * actual_bpar + beta / 2.0 * total_pressure
    assert np.linalg.norm(balance) < tol * np.linalg.norm(total_pressure)
    wrong_ref_beta = -(beta / 2.0) * pressure / (1.0 + beta / 2.0 * susceptibility)
    assert np.linalg.norm(actual_bpar - wrong_ref_beta) > 100 * tol * np.linalg.norm(
        actual_bpar
    )

    external_bpar = np.zeros_like(np.asarray(out.bpar))
    external_bpar[iy, ix] = (0.04 + 0.03j) * (1.0 + 0.2 * np.cos(z))
    zero = jnp.zeros_like(out.phi)
    H_bpar = build_H(
        jnp.zeros_like(jnp.asarray(G, ctype)),
        cache.Jl,
        zero,
        species["tz"],
        bpar=jnp.asarray(external_bpar, ctype),
        JlB=cache.JlB,
    )
    expected_H = direct_jb * external_bpar[iy, ix][None, None, :]
    np.testing.assert_allclose(
        np.asarray(H_bpar[:, :, 0, iy, ix]), expected_H, rtol=tol, atol=tol
    )
    assert np.linalg.norm(expected_H - B[None, None, :] * expected_H) > 100 * tol

    G_flux = np.array(G, copy=True)
    for s in range(2):
        for ell in range(3):
            G_flux[s, ell, 0, iy, ix] = (0.03 + 0.02j * (s + ell + 1)) * np.exp(
                1j * (ell + 1) * z
            )
    channels = particle_flux_channel_species(
        jnp.asarray(G_flux, ctype),
        zero,
        zero,
        jnp.asarray(external_bpar, ctype),
        cache,
        grid,
        params,
        fieldline_quadrature_weights(geom, grid)[1],
        use_dealias=False,
    )
    direct_uB = np.sum(direct_jb * G_flux[:, :, 0, iy, ix], axis=1)
    flux_fac = np.asarray(fieldline_quadrature_weights(geom, grid)[1])
    fac = np.asarray(
        transport_mode_weights(grid.ky, grid.kx.size, ny_full=grid.ny_full)
    )[iy, ix]

    def direct_flux_for(bpar):
        velocity = 1j * float(grid.ky[iy]) * bpar
        return (
            2.0
            * fac
            * np.sum(
                flux_fac[None, :] * np.real(np.conj(velocity)[None, :] * direct_uB),
                axis=1,
            )
            * np.asarray(species["density"] * species["tz"])
        )

    direct_flux = direct_flux_for(external_bpar[iy, ix])
    np.testing.assert_allclose(np.asarray(channels[2]), direct_flux, rtol=tol, atol=tol)
    wrong_flux = direct_flux_for(B * external_bpar[iy, ix])
    assert np.linalg.norm(direct_flux - wrong_flux) > 100 * tol * np.linalg.norm(
        direct_flux
    )


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
    zonal_profile = jnp.asarray(0.2 + 0.05j * z, dtype=G.dtype)
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
        .set(jnp.asarray(0.2 + 0.05j * cache.bmag, dtype=G.dtype))
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
