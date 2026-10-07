"""Tests for modular field solves and custom-VJP behavior."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import eval_laguerre, j1

from gkx.config import CycloneBaseCase, GeometryConfig, GridConfig
from gkx.core_ky_layout import (
    symmetrize_self_conjugate_rows,
    transport_mode_weights,
)
from gkx.geometry import SAlphaGeometry
from gkx.core_grid import build_spectral_grid
from scripts.checks._gates.validation_gates import (
    particle_flux_channel_species,
)
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.moments import build_H, quasineutrality_phi
from gkx.operators.linear.params import LinearParams
from gkx.operators.moments import fieldline_quadrature_weights
from gkx.parallel.velocity_drive import electrostatic_phi_reference
from gkx.terms.assembly import assemble_rhs_terms_cached
from gkx.terms.config import TermConfig
from gkx.terms.fields import _solve_fields_impl, solve_fields
from gkx.terms.linear_terms import linked_streaming_contribution, mirror_contribution
from gkx.core_grid import select_ky_grid
from gkx.terms import assembly as assembly_mod
from gkx.terms.assembly import (
    assemble_rhs_cached,
    assemble_rhs_cached_electrostatic_jit,
    assemble_rhs_cached_jit,
    compute_fields_cached,
)
from gkx.terms.config import FieldState
import gkx.terms as term_pkg
from gkx.core_velocity import hermite_ladder_coeffs
from gkx.operators.linear.streaming import (
    _check_positive,
    abs_z_linked_fft,
    apply_hermite_v,
    apply_hermite_v2,
    apply_laguerre_x,
    grad_z_linked_fft,
    grad_z_periodic,
    shift_axis,
    streaming_ladder_term,
)


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


def _three_field_physical_energy_channels(
    G, fields, cache, params, zweight, jax_values, iy, ix
):
    """Return particle, magnetic, and wrong-B magnetic energy channels."""

    H = build_H(
        G,
        cache.Jl,
        fields.phi,
        jax_values["tz"],
        fields.apar,
        jax_values["vth"],
        fields.bpar,
        cache.JlB,
    )
    nt = jax_values["density"] * jax_values["temp"]
    entropy = 0.5 * jnp.sum(
        zweight[None, None, None, :]
        * nt[:, None, None, None]
        * jnp.abs(H[:, :, :, iy, ix]) ** 2
    )
    boltzmann = 0.5 * jnp.sum(
        zweight
        * jnp.abs(fields.phi[iy, ix]) ** 2
        * jnp.sum(
            jax_values["density"] * jax_values["charge"] ** 2 / jax_values["temp"]
        )
    )

    def field_energy(field, metric):
        return jnp.sum(zweight * metric * jnp.abs(field[iy, ix]) ** 2 / params.beta)

    B2 = cache.bmag**2
    apar_metric = cache.kperp2[iy, ix]
    return (
        entropy - boltzmann,
        field_energy(fields.apar, apar_metric * B2),
        field_energy(fields.bpar, B2),
        field_energy(fields.apar, apar_metric),
        field_energy(fields.bpar, 1.0),
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
    source_args = (
        jl_jax,
        jb_jax,
        jnp.ones(grid.z.size, dtype=dtype),
        values,
        jax_values,
        iy,
        ix,
    )
    source_energy = _three_field_source_quadratic(
        G_jax, flat_cache, params, source_args
    )
    physical_channels = _three_field_physical_energy_channels(
        G_jax, flat_out, flat_cache, params, source_args[2], jax_values, iy, ix
    )
    physical_energy = sum(physical_channels[:3])
    np.testing.assert_allclose(
        np.asarray(source_energy), np.asarray(physical_energy), rtol=tol, atol=tol
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
    energy_errors = []
    energy_mutations = []
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
        source_energy = _three_field_source_quadratic(
            G_jax,
            cache,
            params,
            (jl_jax, jb_jax, independent_vol, values, jax_values, iy, ix),
        )
        physical_channels = _three_field_physical_energy_channels(
            G_jax, fields, cache, params, independent_vol, jax_values, iy, ix
        )
        physical_energy = sum(physical_channels[:3])
        energy_scale = jnp.abs(physical_energy)
        energy_errors.append(
            float(jnp.abs(source_energy - physical_energy) / energy_scale)
        )
        for channel, wrong_channel in zip(
            physical_channels[1:3], physical_channels[3:], strict=True
        ):
            wrong_energy = physical_energy - channel + wrong_channel
            energy_mutations.append(
                float(jnp.abs(source_energy - wrong_energy) / jnp.abs(channel))
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
    assert max(energy_errors) < resolved_tol
    assert min(energy_mutations) > 10 * tol
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


# ---- from test_terms_assembly.py ----


def test_assemble_rhs_terms_sum_matches_total() -> None:
    grid_full = build_spectral_grid(GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.28, Ly=6.28))
    grid = select_ky_grid(grid_full, 1)
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, drift_scale=1.0)
    params = LinearParams(
        fprim=0.8,
        tprim=2.49,
        tprim_e=0.0,
        omega_d_scale=1.0,
        omega_star_scale=1.0,
        rho_star=1.0,
        kpar_scale=float(geom.gradpar()),
        nu=0.0,
        D_hyper=0.07,
    )
    Nl, Nm = 4, 4
    cache = build_linear_cache(grid, geom, params, Nl, Nm)
    rng = np.random.default_rng(0)
    G0 = rng.normal(
        size=(Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size)
    ) + 1j * rng.normal(size=(Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size))
    G0 = jnp.asarray(G0)
    term_cfg = TermConfig(hyperdiffusion=1.0)
    rhs_total, _fields = assemble_rhs_cached(G0, cache, params, terms=term_cfg)
    rhs_terms, _fields_terms, contrib = assemble_rhs_terms_cached(
        G0, cache, params, terms=term_cfg
    )
    rhs_sum = (
        contrib["streaming"]
        + contrib["mirror"]
        + contrib["curvature"]
        + contrib["gradb"]
        + contrib["diamagnetic"]
        + contrib["collisions"]
        + contrib["hypercollisions"]
        + contrib["hyperdiffusion"]
        + contrib["end_damping"]
    )
    assert np.allclose(
        np.asarray(rhs_terms), np.asarray(rhs_total), rtol=1.0e-6, atol=1.0e-8
    )
    assert np.allclose(
        np.asarray(rhs_sum), np.asarray(rhs_total), rtol=1.0e-6, atol=1.0e-8
    )


def test_assemble_rhs_cached_validates_state_shape_and_species_match() -> None:
    grid_full = build_spectral_grid(GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.28, Ly=6.28))
    grid = select_ky_grid(grid_full, 1)
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, drift_scale=1.0)
    params = LinearParams(
        fprim=0.8,
        tprim=2.49,
        tprim_e=0.0,
        omega_d_scale=1.0,
        omega_star_scale=1.0,
        rho_star=1.0,
        kpar_scale=float(geom.gradpar()),
        nu=0.0,
    )
    cache = build_linear_cache(grid, geom, params, 3, 3)

    with pytest.raises(ValueError):
        assemble_rhs_cached(jnp.ones((2, 3, 4, 5), dtype=jnp.complex64), cache, params)

    G_species = jnp.ones(
        (2, 3, 3, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64
    )
    with pytest.raises(ValueError):
        assemble_rhs_cached(G_species, cache, params)


def test_compute_fields_cached_matches_rhs_fields_and_validation() -> None:
    grid_full = build_spectral_grid(GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.28, Ly=6.28))
    grid = select_ky_grid(grid_full, 1)
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, drift_scale=1.0)
    params = LinearParams(
        fprim=0.8,
        tprim=2.49,
        tprim_e=0.0,
        omega_d_scale=1.0,
        omega_star_scale=1.0,
        rho_star=1.0,
        kpar_scale=float(geom.gradpar()),
        nu=0.0,
    )
    cache = build_linear_cache(grid, geom, params, 3, 3)
    rng = np.random.default_rng(2)
    G0 = rng.normal(
        size=(3, 3, grid.ky.size, grid.kx.size, grid.z.size)
    ) + 1j * rng.normal(size=(3, 3, grid.ky.size, grid.kx.size, grid.z.size))
    G0 = jnp.asarray(G0)

    rhs, fields_rhs = assemble_rhs_cached(G0, cache, params, use_custom_vjp=False)
    fields_only = compute_fields_cached(G0, cache, params, use_custom_vjp=False)
    assert rhs.shape == G0.shape
    assert np.allclose(
        np.asarray(fields_only.phi),
        np.asarray(fields_rhs.phi),
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    with pytest.raises(ValueError):
        compute_fields_cached(
            jnp.ones((2, 3, 4, 5), dtype=jnp.complex64), cache, params
        )


def test_disabled_em_fields_skip_hamiltonian_branches(monkeypatch) -> None:
    grid_full = build_spectral_grid(GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.28, Ly=6.28))
    grid = select_ky_grid(grid_full, 1)
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, drift_scale=1.0)
    params = LinearParams(
        fprim=0.8,
        tprim=2.49,
        tprim_e=0.0,
        omega_d_scale=1.0,
        omega_star_scale=1.0,
        rho_star=1.0,
        kpar_scale=float(geom.gradpar()),
        nu=0.0,
        beta=0.0,
    )
    cache = build_linear_cache(grid, geom, params, 3, 3)
    G0 = jnp.ones((3, 3, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)
    terms = TermConfig(apar=0.0, bpar=0.0)
    fields = FieldState(
        phi=jnp.ones((grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64),
        apar=jnp.zeros((grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64),
        bpar=jnp.zeros((grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64),
    )
    apar, bpar, h_apar, h_bpar = assembly_mod._rhs_field_views(fields, terms)
    assert apar.shape == fields.phi.shape
    assert bpar.shape == fields.phi.shape
    assert h_apar is None
    assert h_bpar is None

    seen: dict[str, bool] = {}
    original_build_h = assembly_mod.build_H

    def _record_build_h(*args, **kwargs):
        seen["apar_is_none"] = kwargs.get("apar") is None
        seen["bpar_is_none"] = kwargs.get("bpar") is None
        return original_build_h(*args, **kwargs)

    monkeypatch.setattr(assembly_mod, "build_H", _record_build_h)
    assemble_rhs_cached(G0, cache, params, terms=terms, use_custom_vjp=False)
    assert seen == {"apar_is_none": True, "bpar_is_none": True}


def test_assemble_rhs_cached_jit_accepts_term_config() -> None:
    grid_full = build_spectral_grid(GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.28, Ly=6.28))
    grid = select_ky_grid(grid_full, 1)
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, drift_scale=1.0)
    params = LinearParams(
        fprim=0.8,
        tprim=2.49,
        tprim_e=0.0,
        omega_d_scale=1.0,
        omega_star_scale=1.0,
        rho_star=1.0,
        kpar_scale=float(geom.gradpar()),
        nu=0.0,
        beta=0.0,
    )
    cache = build_linear_cache(grid, geom, params, 3, 3)
    G0 = jnp.ones((3, 3, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)
    rhs, fields = assemble_rhs_cached_jit(
        G0, cache, params, TermConfig(apar=0.0, bpar=0.0)
    )
    assert rhs.shape == G0.shape
    assert fields.phi.shape == (grid.ky.size, grid.kx.size, grid.z.size)


def test_electrostatic_rhs_jit_matches_generic_zero_em_fields() -> None:
    grid_full = build_spectral_grid(GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.28, Ly=6.28))
    grid = select_ky_grid(grid_full, 1)
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, drift_scale=1.0)
    params = LinearParams(
        fprim=0.8,
        tprim=2.49,
        tprim_e=0.0,
        omega_d_scale=1.0,
        omega_star_scale=1.0,
        rho_star=1.0,
        kpar_scale=float(geom.gradpar()),
        nu=0.0,
        beta=0.0,
    )
    cache = build_linear_cache(grid, geom, params, 3, 4)
    rng = np.random.default_rng(8)
    G0 = rng.normal(
        size=(3, 4, grid.ky.size, grid.kx.size, grid.z.size)
    ) + 1j * rng.normal(size=(3, 4, grid.ky.size, grid.kx.size, grid.z.size))
    G0 = jnp.asarray(G0, dtype=jnp.complex64)
    terms = TermConfig(apar=0.0, bpar=0.0)

    rhs_generic, fields_generic = assemble_rhs_cached_jit(G0, cache, params, terms)
    rhs_electrostatic, fields_electrostatic = assemble_rhs_cached_electrostatic_jit(
        G0, cache, params, terms
    )

    np.testing.assert_allclose(
        np.asarray(rhs_electrostatic), np.asarray(rhs_generic), rtol=1.0e-6, atol=1.0e-6
    )
    np.testing.assert_allclose(
        np.asarray(fields_electrostatic.phi),
        np.asarray(fields_generic.phi),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_external_phi_source_shifts_fields_and_rhs() -> None:
    grid_full = build_spectral_grid(GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.28, Ly=6.28))
    grid = select_ky_grid(grid_full, 1)
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, drift_scale=1.0)
    params = LinearParams(
        fprim=0.0,
        tprim=0.0,
        tprim_e=0.0,
        omega_d_scale=1.0,
        omega_star_scale=1.0,
        rho_star=1.0,
        kpar_scale=float(geom.gradpar()),
        nu=0.0,
    )
    cache = build_linear_cache(grid, geom, params, 2, 2)
    G0 = jnp.zeros((2, 2, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)

    fields0 = compute_fields_cached(G0, cache, params, use_custom_vjp=False)
    fields_src = compute_fields_cached(
        G0, cache, params, use_custom_vjp=False, external_phi=0.25
    )
    np.testing.assert_allclose(
        np.asarray(fields_src.phi - fields0.phi), 0.25, atol=1.0e-7
    )

    rhs0, _ = assemble_rhs_cached(G0, cache, params, use_custom_vjp=False)
    rhs_src, _ = assemble_rhs_cached(
        G0, cache, params, use_custom_vjp=False, external_phi=0.25
    )
    assert not np.allclose(np.asarray(rhs_src), np.asarray(rhs0))


def test_collision_zero_guard_uses_current_nu_not_cache_build_nu() -> None:
    grid_full = build_spectral_grid(GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.28, Ly=6.28))
    grid = select_ky_grid(grid_full, 1)
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, drift_scale=1.0)
    params = LinearParams(
        fprim=0.0,
        tprim=0.0,
        tprim_e=0.0,
        omega_d_scale=1.0,
        omega_star_scale=1.0,
        rho_star=1.0,
        kpar_scale=float(geom.gradpar()),
        nu=0.0,
    )
    cache = build_linear_cache(grid, geom, params, 3, 3)
    rng = np.random.default_rng(3)
    G0 = rng.normal(
        size=(3, 3, grid.ky.size, grid.kx.size, grid.z.size)
    ) + 1j * rng.normal(size=(3, 3, grid.ky.size, grid.kx.size, grid.z.size))
    G0 = jnp.asarray(G0, dtype=jnp.complex64)
    terms = TermConfig(
        streaming=0.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=1.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )

    rhs_zero, _fields_zero, contrib_zero = assemble_rhs_terms_cached(
        G0,
        cache,
        params,
        terms=terms,
        use_custom_vjp=False,
    )
    np.testing.assert_allclose(np.asarray(rhs_zero), 0.0, atol=1.0e-7)
    np.testing.assert_allclose(np.asarray(contrib_zero["collisions"]), 0.0, atol=1.0e-7)

    rhs_nonzero, _fields_nonzero, contrib_nonzero = assemble_rhs_terms_cached(
        G0,
        cache,
        replace(params, nu=0.2),
        terms=terms,
        use_custom_vjp=False,
    )
    assert np.linalg.norm(np.asarray(rhs_nonzero)) > 1.0e-5
    assert np.linalg.norm(np.asarray(contrib_nonzero["collisions"])) > 1.0e-5


def test_collision_zero_guard_preserves_preexpanded_collision_operator() -> None:
    grid_full = build_spectral_grid(GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.28, Ly=6.28))
    grid = select_ky_grid(grid_full, 1)
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, drift_scale=1.0)
    params = LinearParams(
        fprim=0.0,
        tprim=0.0,
        tprim_e=0.0,
        omega_d_scale=1.0,
        omega_star_scale=1.0,
        rho_star=1.0,
        kpar_scale=float(geom.gradpar()),
        nu=0.0,
    )
    cache = build_linear_cache(grid, geom, params, 3, 3)
    rng = np.random.default_rng(4)
    G0 = rng.normal(
        size=(3, 3, grid.ky.size, grid.kx.size, grid.z.size)
    ) + 1j * rng.normal(size=(3, 3, grid.ky.size, grid.kx.size, grid.z.size))
    G0 = jnp.asarray(G0, dtype=jnp.complex64)
    cache_with_collision_matrix = replace(
        cache,
        collision_lam=jnp.ones_like(G0[None, ...], dtype=jnp.float32) * 0.2,
    )
    terms = TermConfig(
        streaming=0.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=1.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )

    rhs, _fields, contrib = assemble_rhs_terms_cached(
        G0,
        cache_with_collision_matrix,
        params,
        terms=terms,
        use_custom_vjp=False,
    )
    assert np.linalg.norm(np.asarray(rhs)) > 1.0e-5
    assert np.linalg.norm(np.asarray(contrib["collisions"])) > 1.0e-5


def test_collision_zero_weight_skips_invalid_preexpanded_operator_shape() -> None:
    grid_full = build_spectral_grid(GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.28, Ly=6.28))
    grid = select_ky_grid(grid_full, 1)
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, drift_scale=1.0)
    params = LinearParams(
        fprim=0.0,
        tprim=0.0,
        tprim_e=0.0,
        omega_d_scale=1.0,
        omega_star_scale=1.0,
        rho_star=1.0,
        kpar_scale=float(geom.gradpar()),
        nu=0.0,
    )
    cache = build_linear_cache(grid, geom, params, 3, 3)
    cache_with_unused_bad_collision_matrix = replace(
        cache,
        collision_lam=jnp.ones_like(cache.lb_lam, dtype=jnp.float32),
    )
    rng = np.random.default_rng(5)
    G0 = rng.normal(
        size=(3, 3, grid.ky.size, grid.kx.size, grid.z.size)
    ) + 1j * rng.normal(size=(3, 3, grid.ky.size, grid.kx.size, grid.z.size))
    G0 = jnp.asarray(G0, dtype=jnp.complex64)
    terms = TermConfig(
        streaming=0.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )

    rhs, _fields, contrib = assemble_rhs_terms_cached(
        G0,
        cache_with_unused_bad_collision_matrix,
        params,
        terms=terms,
        use_custom_vjp=False,
    )
    np.testing.assert_allclose(np.asarray(rhs), 0.0, atol=1.0e-7)
    np.testing.assert_allclose(np.asarray(contrib["collisions"]), 0.0, atol=1.0e-7)


def test_static_zero_switches_stay_static_inside_a_trace() -> None:
    """A term switch the host can see is off stays off under ``jit``.

    ``_is_static_zero`` used to ask whether a ``jnp`` copy of its argument was
    traced. Inside a trace that copy always is, so every switch of every jitted
    run answered "not statically zero" and every fast path this predicate
    selects was silently abandoned -- including the electrostatic RHS for a
    ``TermConfig`` with ``apar = bpar = 0``, which is what the linear
    benchmarks and every electrostatic nonlinear run are.
    """

    import jax

    from gkx.operators.nonlinear.rhs import linear_rhs_jit_for_terms_impl

    seen: dict[str, object] = {}

    def probe(x: jnp.ndarray) -> jnp.ndarray:
        seen["python_zero"] = assembly_mod._is_static_zero(0.0)
        seen["python_one"] = assembly_mod._is_static_zero(1.0)
        seen["host_array"] = assembly_mod._is_static_zero(np.zeros(3))
        seen["underflows"] = assembly_mod._is_static_zero(1.0e-50, jnp.float32)
        seen["traced"] = assembly_mod._is_static_zero(x)
        seen["route"] = linear_rhs_jit_for_terms_impl(TermConfig(apar=0.0, bpar=0.0))
        return x * 2.0

    jax.jit(probe)(jnp.asarray(3.0))
    assert seen["python_zero"] is True
    assert seen["python_one"] is False
    assert seen["host_array"] is True
    # ``dtype`` still reproduces the operator's own cast.
    assert seen["underflows"] is True
    # A value that really is traced stays unknowable, which is the only case
    # the predicate was ever entitled to give up on.
    assert seen["traced"] is False
    assert seen["route"] is assemble_rhs_cached_electrostatic_jit
    assert (
        linear_rhs_jit_for_terms_impl(TermConfig(apar=1.0, bpar=0.0))
        is assemble_rhs_cached_jit
    )


def _linked_pilot_rhs_case(**param_overrides):
    """Nx=8/Ny=16 linked Cyclone case with four chain lengths and |kz| hypercollisions."""

    from gkx.config import CycloneBaseCase
    from gkx.geometry import ensure_flux_tube_geometry_data

    grid_cfg = GridConfig(
        Nx=8, Ny=16, Nz=8, Lx=6.28, Ly=6.28, boundary="linked", jtwist=1
    )
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = ensure_flux_tube_geometry_data(
        SAlphaGeometry.from_config(cfg.geometry), grid.z
    )
    options = dict(
        fprim=0.8,
        tprim=2.49,
        kpar_scale=float(geom.gradpar()),
        nu=0.0,
        nu_hyper_m=1.0,
        hypercollisions_kz=1.0,
    )
    options.update(param_overrides)
    params = LinearParams(**options)
    Nl, Nm = 2, 4
    cache = build_linear_cache(grid, geom, params, Nl, Nm)
    rng = np.random.default_rng(9)
    shape = (Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size)
    G0 = jnp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape))
    return cache, params, G0


def _shared_route_calls(monkeypatch):
    calls: list[bool] = []
    shared = assembly_mod._shared_linked_streaming_hypercollisions

    def counting(*args, **kwargs):
        result = shared(*args, **kwargs)
        calls.append(result is not None)
        return result

    monkeypatch.setattr(
        assembly_mod, "_shared_linked_streaming_hypercollisions", counting
    )
    return calls


def test_linked_streaming_and_hypercollisions_share_the_chain_transform(
    monkeypatch,
) -> None:
    """Q9: one stacked transform per chain class gives the separate terms.

    Every named term, the total and the state VJP match the separate
    streaming and hypercollision transforms (measured bitwise for the primal
    terms on XLA:CPU; the tolerance leaves room for FFT batch-layout roundoff).
    """

    import jax

    cache, params, G0 = _linked_pilot_rhs_case()
    assert len(cache.linked_indices) >= 3
    term_cfg = TermConfig(hypercollisions=1.0)
    cotangent = jnp.conj(G0[::-1])

    def run():
        total, _fields, contrib = assemble_rhs_terms_cached(
            G0, cache, params, terms=term_cfg
        )
        grad = jax.grad(
            lambda g: jnp.real(
                jnp.vdot(
                    cotangent, assemble_rhs_cached(g, cache, params, terms=term_cfg)[0]
                )
            )
        )(G0)
        return total, contrib, grad

    calls = _shared_route_calls(monkeypatch)
    total, contrib, grad = run()
    assert calls and all(calls)
    monkeypatch.setattr(
        assembly_mod,
        "_shared_linked_streaming_hypercollisions",
        lambda *args, **kwargs: None,
    )
    total_ref, contrib_ref, grad_ref = run()
    assert float(jnp.linalg.norm(contrib_ref["hypercollisions"])) > 0.0
    for key in contrib_ref:
        np.testing.assert_allclose(
            np.asarray(contrib[key]), np.asarray(contrib_ref[key]), rtol=1e-6, atol=1e-6
        )
    np.testing.assert_allclose(
        np.asarray(total), np.asarray(total_ref), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(grad), np.asarray(grad_ref), rtol=1e-6, atol=1e-5
    )


@pytest.mark.parametrize(
    "case",
    ["hypercollision_weight_zero", "kz_branch_off", "streaming_off", "periodic"],
)
def test_shared_linked_transform_falls_back_when_a_term_is_static_off(
    monkeypatch, case
) -> None:
    if case == "periodic":
        grid = build_spectral_grid(GridConfig(Nx=4, Ny=4, Nz=8, Lx=6.28, Ly=6.28))
        geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778)
        params = LinearParams(hypercollisions_kz=1.0, kpar_scale=float(geom.gradpar()))
        cache = build_linear_cache(grid, geom, params, 2, 4)
        shape = (2, 4, grid.ky.size, grid.kx.size, grid.z.size)
        G0 = jnp.ones(shape, dtype=jnp.complex64)
    else:
        overrides = {"kz_branch_off": dict(hypercollisions_kz=0.0)}.get(case, {})
        cache, params, G0 = _linked_pilot_rhs_case(**overrides)
    term_cfg = {
        "hypercollision_weight_zero": TermConfig(hypercollisions=0.0),
        "streaming_off": TermConfig(hypercollisions=1.0, streaming=0.0),
    }.get(case, TermConfig(hypercollisions=1.0))
    calls = _shared_route_calls(monkeypatch)
    assemble_rhs_terms_cached(G0, cache, params, terms=term_cfg)
    assert calls == [False]


# ---- from test_linear_streaming.py ----
# Unit tests for low-level term operators.


def test_fft_z_operators_preserve_complex64_with_float64_wavenumbers() -> None:
    """FFT multipliers must not promote complex64 states under x64/sharding."""

    nz = 8
    z = jnp.linspace(0.0, 2.0 * jnp.pi, nz, endpoint=False)
    dz = z[1] - z[0]
    f = jnp.exp(1j * z).astype(jnp.complex64)
    kz = jnp.asarray(2.0 * jnp.pi * jnp.fft.fftfreq(nz, d=dz), dtype=jnp.float64)

    out_periodic = grad_z_periodic(f, kz=kz)

    linked_f = jnp.stack([f, 0.5j * f], axis=0)[None, ...]
    idx_map = jnp.asarray([[0, 1]], dtype=jnp.int32)
    kz_link = jnp.asarray(
        2.0 * jnp.pi * jnp.fft.fftfreq(2 * nz, d=dz), dtype=jnp.float64
    )
    out_linked = grad_z_linked_fft(
        linked_f, dz=dz, linked_indices=(idx_map,), linked_kz=(kz_link,)
    )
    out_abs = abs_z_linked_fft(
        linked_f, linked_indices=(idx_map,), linked_kz=(kz_link,)
    )

    assert out_periodic.dtype == jnp.complex64
    assert out_linked.dtype == jnp.complex64
    assert out_abs.dtype == jnp.complex64


def test_grad_z_periodic_requires_dz_or_kz() -> None:
    with pytest.raises(ValueError):
        grad_z_periodic(jnp.ones((8,)))


@pytest.mark.parametrize("nlinks,nz", [(1, 7), (1, 8), (3, 3), (2, 4)])
@pytest.mark.parametrize("linked", [False, True])
def test_fft_highest_modes_and_ad_contract(nlinks, nz, linked) -> None:
    """Analytic DFT eigenvalues and AD; even Nyquist uses -N/2, unlike GX."""
    n = nlinks * nz
    dz = 0.3
    modes = (-(n // 2), (n - 1) // 2)
    kz = 2 * jnp.pi * jnp.fft.fftfreq(n, d=dz)
    order = jnp.arange(nlinks - 1, -1, -1)

    def derivative(value):
        if linked:
            return grad_z_linked_fft(
                value, dz=dz, linked_indices=(order[None, :],), linked_kz=(kz,)
            )
        return grad_z_periodic(value, dz=dz)

    for mode in modes:
        wave = jnp.exp(2j * jnp.pi * mode * jnp.arange(n) / n)
        if linked:
            wave = wave.reshape(nlinks, nz)[order][None, ...]
        expected = (2j * jnp.pi * mode / (n * dz)) * wave
        actual, tangent = jax.jvp(jax.jit(derivative), (wave,), (wave,))
        if not linked:
            assert jnp.allclose(grad_z_periodic(wave, kz=kz), actual, atol=3e-5)
        assert jnp.allclose(actual, expected, rtol=3e-5, atol=3e-5)
        assert jnp.allclose(tangent, expected, rtol=3e-5, atol=3e-5)
        # Real parameter pullback avoids ambiguous complex-gradient conventions.
        gradient = jax.grad(
            lambda amplitude: jnp.real(jnp.vdot(expected, derivative(amplitude * wave)))
        )(1.0)
        assert jnp.allclose(gradient, jnp.real(jnp.vdot(expected, expected)), rtol=3e-5)


def test_grad_z_linked_fft_with_inverse_permutation_matches_scatter_path() -> None:
    ny, nx, nz = 1, 2, 8
    z = jnp.linspace(0.0, 2.0 * jnp.pi, nz, endpoint=False)
    dz = z[1] - z[0]
    f = jnp.zeros((ny, nx, nz), dtype=jnp.complex64)
    f = f.at[0, 0, :].set(jnp.exp(1j * z))
    f = f.at[0, 1, :].set(2.0 * jnp.exp(1j * 2.0 * z))

    idx_map = jnp.asarray([[1, 0]], dtype=jnp.int32)
    kz_link = 2.0 * jnp.pi * jnp.fft.fftfreq(2 * nz, d=dz)
    inv = jnp.asarray([1, 0], dtype=jnp.int32)

    out_scatter = grad_z_linked_fft(
        f,
        dz=dz,
        linked_indices=(idx_map,),
        linked_kz=(kz_link,),
    )
    out_perm = grad_z_linked_fft(
        f,
        dz=dz,
        linked_indices=(idx_map,),
        linked_kz=(kz_link,),
        linked_inverse_permutation=inv,
        linked_full_cover=True,
    )
    assert jnp.allclose(out_perm, out_scatter, atol=1.0e-5)


def test_linked_fft_gather_paths_match_scatter_for_derivative_and_abs() -> None:
    ny, nx, nz = 1, 2, 8
    z = jnp.linspace(0.0, 2.0 * jnp.pi, nz, endpoint=False)
    dz = z[1] - z[0]
    f = jnp.zeros((ny, nx, nz), dtype=jnp.complex64)
    f = f.at[0, 0, :].set(jnp.exp(1j * z))
    f = f.at[0, 1, :].set((0.5 + 0.25j) * jnp.exp(1j * 2.0 * z))
    idx_map = jnp.asarray([[0, 1]], dtype=jnp.int32)
    kz_link = 2.0 * jnp.pi * jnp.fft.fftfreq(2 * nz, d=dz)
    gather_map = jnp.asarray([0, 1], dtype=jnp.int32)
    gather_mask = jnp.asarray([True, True])

    grad_scatter = grad_z_linked_fft(
        f, dz=dz, linked_indices=(idx_map,), linked_kz=(kz_link,)
    )
    grad_gather = grad_z_linked_fft(
        f,
        dz=dz,
        linked_indices=(idx_map,),
        linked_kz=(kz_link,),
        linked_gather_map=gather_map,
        linked_gather_mask=gather_mask,
        linked_use_gather=True,
    )
    abs_scatter = abs_z_linked_fft(f, linked_indices=(idx_map,), linked_kz=(kz_link,))
    abs_gather = abs_z_linked_fft(
        f,
        linked_indices=(idx_map,),
        linked_kz=(kz_link,),
        linked_gather_map=gather_map,
        linked_gather_mask=gather_mask,
        linked_use_gather=True,
    )

    assert jnp.allclose(grad_gather, grad_scatter, atol=1.0e-5)
    assert jnp.allclose(abs_gather, abs_scatter, atol=1.0e-5)


def test_grad_z_linked_fft_restores_negative_ky_rows_by_conjugate_symmetry() -> None:
    ny, nx, nz = 8, 4, 8
    z = jnp.linspace(0.0, 2.0 * jnp.pi, nz, endpoint=False)
    dz = z[1] - z[0]
    f = jnp.zeros((ny, nx, nz), dtype=jnp.complex64)
    f = f.at[1, 0, :].set(jnp.exp(1j * z))
    f = f.at[1, 1, :].set((1.0 - 0.5j) * jnp.exp(1j * 2.0 * z))
    kx_neg = jnp.asarray([0, 3, 2, 1], dtype=jnp.int32)
    f = f.at[7, :, :].set(jnp.conj(jnp.take(f[1], kx_neg, axis=0)))

    idx_map = jnp.asarray([[1, 1 + ny]], dtype=jnp.int32)
    kz_link = 2.0 * jnp.pi * jnp.fft.fftfreq(2 * nz, d=dz)
    out = grad_z_linked_fft(
        f,
        dz=dz,
        linked_indices=(idx_map,),
        linked_kz=(kz_link,),
    )

    assert jnp.max(jnp.abs(out[7])) > 0.0
    assert jnp.allclose(out[7], jnp.conj(jnp.take(out[1], kx_neg, axis=0)), atol=1.0e-5)


def test_abs_z_linked_fft_restores_negative_ky_rows_by_conjugate_symmetry() -> None:
    ny, nx, nz = 8, 4, 8
    z = jnp.linspace(0.0, 2.0 * jnp.pi, nz, endpoint=False)
    f = jnp.zeros((ny, nx, nz), dtype=jnp.complex64)
    f = f.at[1, 0, :].set(jnp.exp(1j * z))
    f = f.at[1, 1, :].set((1.0 + 0.25j) * jnp.exp(1j * 2.0 * z))
    kx_neg = jnp.asarray([0, 3, 2, 1], dtype=jnp.int32)
    f = f.at[7, :, :].set(jnp.conj(jnp.take(f[1], kx_neg, axis=0)))

    idx_map = jnp.asarray([[1, 1 + ny]], dtype=jnp.int32)
    kz_link = 2.0 * jnp.pi * jnp.fft.fftfreq(2 * nz, d=z[1] - z[0])
    out = abs_z_linked_fft(
        f,
        linked_indices=(idx_map,),
        linked_kz=(kz_link,),
    )

    assert jnp.max(jnp.abs(out[7])) > 0.0
    assert jnp.allclose(out[7], jnp.conj(jnp.take(out[1], kx_neg, axis=0)), atol=1.0e-5)


def test_linked_fft_validates_inputs() -> None:
    f = jnp.ones((1, 2, 8), dtype=jnp.complex64)
    dz = jnp.asarray(0.1)
    kz = 2.0 * jnp.pi * jnp.fft.fftfreq(16, d=dz)
    with pytest.raises(ValueError):
        grad_z_linked_fft(f, dz=dz, linked_indices=(), linked_kz=())
    with pytest.raises(ValueError):
        grad_z_linked_fft(
            f,
            dz=dz,
            linked_indices=(jnp.asarray([[0, 1]], dtype=jnp.int32),),
            linked_kz=(),
        )
    with pytest.raises(ValueError):
        grad_z_linked_fft(
            f,
            dz=dz,
            linked_indices=(jnp.asarray([0, 1], dtype=jnp.int32),),
            linked_kz=(kz,),
        )
    with pytest.raises(ValueError):
        grad_z_linked_fft(
            f,
            dz=dz,
            linked_indices=(jnp.asarray([[0, 1]], dtype=jnp.int32),),
            linked_kz=(kz,),
            linked_full_cover=True,
        )
    with pytest.raises(ValueError):
        abs_z_linked_fft(f, linked_indices=(), linked_kz=())
    with pytest.raises(ValueError):
        abs_z_linked_fft(
            f,
            linked_indices=(jnp.asarray([[0, 1]], dtype=jnp.int32),),
            linked_kz=(),
        )
    with pytest.raises(ValueError):
        abs_z_linked_fft(
            f,
            linked_indices=(jnp.asarray([0, 1], dtype=jnp.int32),),
            linked_kz=(kz,),
        )
    with pytest.raises(ValueError):
        abs_z_linked_fft(
            f,
            linked_indices=(jnp.asarray([[0, 1]], dtype=jnp.int32),),
            linked_kz=(kz,),
            linked_full_cover=True,
        )


def test_shift_axis_edge_cases() -> None:
    arr = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    assert jnp.allclose(shift_axis(arr, 0, axis=0), arr)
    assert jnp.allclose(shift_axis(arr, 1, axis=0), jnp.asarray([2.0, 3.0, 4.0, 0.0]))
    assert jnp.allclose(shift_axis(arr, -1, axis=0), jnp.asarray([0.0, 1.0, 2.0, 3.0]))
    assert jnp.allclose(shift_axis(arr, 10, axis=0), jnp.zeros_like(arr))
    assert jnp.allclose(shift_axis(arr, -10, axis=0), jnp.zeros_like(arr))


def test_hermite_laguerre_operators_shapes_and_values() -> None:
    G = jnp.zeros((2, 3, 4, 1, 1, 1))
    G = G.at[0, 1, 2, 0, 0, 0].set(1.0)
    hv = apply_hermite_v(G)
    hv2 = apply_hermite_v2(G)
    lx = apply_laguerre_x(G)
    assert hv.shape == G.shape
    assert hv2.shape == G.shape
    assert lx.shape == G.shape
    assert jnp.isfinite(hv).all()
    assert jnp.isfinite(hv2).all()
    assert jnp.isfinite(lx).all()


def test_streaming_term_periodic_and_linked_paths() -> None:
    ns, nl, nm, ny, nx, nz = 1, 2, 3, 1, 2, 8
    z = jnp.linspace(0.0, 2.0 * jnp.pi, nz, endpoint=False)
    dz = z[1] - z[0]
    kz = 2.0 * jnp.pi * jnp.fft.fftfreq(nz, d=dz)
    H = jnp.zeros((ns, nl, nm, ny, nx, nz), dtype=jnp.complex64)
    H = H.at[0, 0, 0, 0, 0, :].set(jnp.exp(1j * z))
    sqrt_p, sqrt_m = hermite_ladder_coeffs(nm - 1)
    sqrt_p = sqrt_p[:nm].reshape((1, 1, nm, 1, 1, 1))
    sqrt_m = sqrt_m[:nm].reshape((1, 1, nm, 1, 1, 1))
    vth = jnp.ones((1, 1, 1, 1, 1, 1), dtype=jnp.float32)

    out_periodic = streaming_ladder_term(
        H, kz=kz, vth=vth, sqrt_p=sqrt_p, sqrt_m=sqrt_m
    )
    assert out_periodic.shape == H.shape
    assert jnp.isfinite(out_periodic).all()

    idx_map = jnp.asarray([[0, 1]], dtype=jnp.int32)
    kz_link = 2.0 * jnp.pi * jnp.fft.fftfreq(2 * nz, d=dz)
    out_linked = streaming_ladder_term(
        H,
        kz=kz,
        vth=vth,
        sqrt_p=sqrt_p,
        sqrt_m=sqrt_m,
        dz=dz,
        use_twist_shift=True,
        linked_indices=(idx_map,),
        linked_kz=(kz_link,),
    )
    assert out_linked.shape == H.shape
    assert jnp.isfinite(out_linked).all()


def test_streaming_term_linked_fd_and_errors() -> None:
    ns, nl, nm, ny, nx, nz = 1, 1, 3, 1, 2, 8
    H = jnp.ones((ns, nl, nm, ny, nx, nz), dtype=jnp.complex64)
    dz = jnp.asarray(0.2)
    kz = 2.0 * jnp.pi * jnp.fft.fftfreq(nz, d=dz)
    sqrt_p, sqrt_m = hermite_ladder_coeffs(nm - 1)
    sqrt_p = sqrt_p[:nm].reshape((1, 1, nm, 1, 1, 1))
    sqrt_m = sqrt_m[:nm].reshape((1, 1, nm, 1, 1, 1))
    vth = jnp.ones((1, 1, 1, 1, 1, 1), dtype=jnp.float32)
    kx_link_plus = jnp.asarray([[1, 0]], dtype=jnp.int32)
    kx_link_minus = jnp.asarray([[1, 0]], dtype=jnp.int32)
    kx_mask = jnp.asarray([[True, True]])
    out = streaming_ladder_term(
        H,
        kz=kz,
        vth=vth,
        sqrt_p=sqrt_p,
        sqrt_m=sqrt_m,
        dz=dz,
        use_twist_shift=True,
        kx_link_plus=kx_link_plus,
        kx_link_minus=kx_link_minus,
        kx_mask_plus=kx_mask,
        kx_mask_minus=kx_mask,
    )
    assert out.shape == H.shape
    assert jnp.isfinite(out).all()

    with pytest.raises(ValueError):
        streaming_ladder_term(
            H, kz=kz, vth=vth, sqrt_p=sqrt_p, sqrt_m=sqrt_m, use_twist_shift=True
        )
    with pytest.raises(ValueError):
        streaming_ladder_term(
            H,
            kz=kz,
            vth=vth,
            sqrt_p=sqrt_p,
            sqrt_m=sqrt_m,
            dz=dz,
            use_twist_shift=True,
        )
    with pytest.raises(ValueError):
        streaming_ladder_term(
            H,
            kz=kz,
            vth=vth,
            sqrt_p=sqrt_p,
            sqrt_m=sqrt_m,
            dz=dz,
            use_twist_shift=True,
            kx_link_plus=kx_link_plus,
            kx_link_minus=kx_link_minus,
        )


def test_terms_positive_validation_checks() -> None:
    _check_positive(1.0, "x")
    _check_positive(jnp.asarray([1.0, 2.0]), "arr")
    with pytest.raises(ValueError):
        _check_positive(0.0, "x")
    with pytest.raises(ValueError):
        _check_positive(jnp.asarray([1.0, 0.0]), "arr")


def test_terms_positive_validation_skips_tracer_runtime_checks() -> None:
    @jax.jit
    def f(x: jnp.ndarray) -> jnp.ndarray:
        _check_positive(x, "x")
        return x + 1.0

    out = f(jnp.asarray(0.0))
    assert float(out) == 1.0


def test_streaming_positive_validation_is_the_shared_guard() -> None:
    """Streaming validates a concrete ``dz`` under ``jit`` like everything else.

    The streaming kernels carried their own copy of the guard, and that copy
    asked whether a ``jnp`` round trip of its argument was traced -- true of
    every argument inside a trace -- so ``dz`` and ``vth`` went unchecked in
    every jitted run. One guard now answers for the whole linear operator.
    """

    from gkx.operators.linear import params as linear_params
    from gkx.operators.linear import streaming as streaming_mod

    assert streaming_mod._check_positive is linear_params._check_positive

    seen: dict[str, str | None] = {}

    def probe(x: jnp.ndarray) -> jnp.ndarray:
        try:
            _check_positive(0.0, "dz")
            seen["dz"] = None
        except ValueError as exc:
            seen["dz"] = str(exc)
        return x

    jax.jit(probe)(jnp.asarray(1.0))
    assert seen["dz"] == "dz must be > 0"


def test_terms_package_lazy_exports() -> None:
    assert callable(term_pkg.assemble_rhs_cached)
    assert callable(term_pkg.assemble_rhs_cached_jit)
    with pytest.raises(AttributeError):
        _ = term_pkg.not_a_real_symbol


def test_terms_config_pytrees_roundtrip() -> None:
    cfg = TermConfig(streaming=0.5, nonlinear=0.25, bpar=0.0)
    leaves, treedef = jax.tree_util.tree_flatten(cfg)
    cfg_rt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert cfg_rt == cfg

    state = FieldState(phi=jnp.ones((2, 2)), apar=jnp.zeros((2, 2)), bpar=None)
    leaves_s, tree_s = jax.tree_util.tree_flatten(state)
    state_rt = jax.tree_util.tree_unflatten(tree_s, leaves_s)
    assert jnp.allclose(state_rt.phi, state.phi)
    assert jnp.allclose(state_rt.apar, state.apar)
    assert state_rt.bpar is None


# --- Q9 (plan 5.3 N2): stacked operands share one transform per chain class ---

_CHAIN_MIXES = (
    ((3, 1),),
    ((2, 1), (1, 2)),
    ((4, 1), (2, 2), (1, 3), (1, 5)),
)


def _chain_mix_maps(classes, *, ny: int, nx: int, nz: int, dz: float):
    """Chain maps ``ky + ny * kx`` over distinct modes, one class per length."""

    import numpy as np

    modes = iter(ky + ny * kx for kx in range(nx) for ky in range(ny))
    indices = tuple(
        np.asarray(
            [[next(modes) for _ in range(nlinks)] for _ in range(nchains)], np.int32
        )
        for nchains, nlinks in classes
    )
    kz = tuple(2.0 * np.pi * np.fft.fftfreq(nlinks * nz, d=dz) for _, nlinks in classes)
    return indices, kz


def _per_chain_reference(f, indices, *, nz: int, dz: float, operator: str):
    """One numpy FFT per chain, of that chain's own length ``nLinks * nz``."""

    import numpy as np

    f = np.asarray(f)
    ny = f.shape[-3]
    out = np.zeros_like(f)
    for idx in indices:
        for chain in idx:
            rows = [(int(m) % ny, int(m) // ny) for m in chain]
            signal = np.concatenate([f[..., y, x, :] for y, x in rows], axis=-1)
            k = 2.0 * np.pi * np.fft.fftfreq(signal.shape[-1], d=dz)
            multiplier = 1j * k if operator == "grad" else np.abs(k)
            result = np.fft.ifft(multiplier * np.fft.fft(signal, axis=-1), axis=-1)
            for link, (y, x) in enumerate(rows):
                out[..., y, x, :] = result[..., link * nz : (link + 1) * nz]
    return out


@pytest.mark.parametrize("classes", _CHAIN_MIXES)
@pytest.mark.parametrize("route", ["gather", "full_cover", "scatter"])
def test_stacked_linked_fft_matches_per_class_operators(classes, route) -> None:
    """Slot i of the stacked call is the per-class operator i on operand i.

    Ny=2 keeps the conjugate restore out of the reference (row 1 is its own
    partner); the full-cover route needs every mode in a chain, the others
    leave three modes outside all chains, which must come back zero.
    """

    import numpy as np

    from gkx.operators.linear.cache_builder import _linked_fft_gather_metadata
    from gkx.operators.linear.streaming import _linked_fft_apply

    ny, nz, dz = 2, 4, 0.3
    n_chain_modes = sum(nchains * nlinks for nchains, nlinks in classes)
    nx = -(-n_chain_modes // ny) if route == "full_cover" else n_chain_modes // ny + 2
    if route == "full_cover" and n_chain_modes % ny:
        pytest.skip("full cover needs an even mode count on Ny=2")
    indices, kz = _chain_mix_maps(classes, ny=ny, nx=nx, nz=nz, dz=dz)
    inverse, full_cover, gather_map, gather_mask, use_gather = (
        _linked_fft_gather_metadata(indices, n_modes=ny * nx)
    )
    options = {
        "gather": dict(
            linked_gather_map=gather_map,
            linked_gather_mask=gather_mask,
            linked_use_gather=use_gather,
        ),
        "full_cover": dict(
            linked_inverse_permutation=inverse, linked_full_cover=full_cover
        ),
        "scatter": {},
    }[route]
    if route == "full_cover":
        assert full_cover
    rng = np.random.default_rng(len(classes))
    shape = (1, 2, 3, ny, nx, nz)
    f = jnp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape))
    g = jnp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape))
    kz_dev = tuple(jnp.asarray(k, dtype=jnp.real(f).dtype) for k in kz)

    stacked = _linked_fft_apply(
        (f, g), indices, kz_dev, operator=("grad", "abs"), **options
    )
    swapped = _linked_fft_apply(
        (g, f), indices, kz_dev, operator=("abs", "grad"), **options
    )
    assert stacked.shape == (2, *shape)
    for slot, (operand, operator) in enumerate(((f, "grad"), (g, "abs"))):
        single = _linked_fft_apply(
            operand, indices, kz_dev, operator=operator, **options
        )
        reference = _per_chain_reference(
            operand, indices, nz=nz, dz=dz, operator=operator
        )
        scale = float(np.max(np.abs(reference)))
        np.testing.assert_allclose(
            np.asarray(stacked[slot]), np.asarray(single), rtol=1e-6, atol=1e-6 * scale
        )
        np.testing.assert_allclose(
            np.asarray(swapped[1 - slot]),
            np.asarray(single),
            rtol=1e-6,
            atol=1e-6 * scale,
        )
        np.testing.assert_allclose(
            np.asarray(single), reference, rtol=1e-5, atol=1e-5 * scale
        )


def test_stacked_linked_fft_validates_operands() -> None:
    from gkx.operators.linear.streaming import _linked_fft_apply

    idx = (jnp.asarray([[0, 1]], dtype=jnp.int32),)
    kz = (2.0 * jnp.pi * jnp.fft.fftfreq(8, d=0.3),)
    f = jnp.zeros((1, 2, 4), dtype=jnp.complex64)
    with pytest.raises(ValueError, match="one operator each"):
        _linked_fft_apply((f, f), idx, kz, operator="grad")
    with pytest.raises(ValueError, match="one operator each"):
        _linked_fft_apply((f, f), idx, kz, operator=("grad",))
    with pytest.raises(ValueError, match="share shape and dtype"):
        _linked_fft_apply((f, f[..., :2]), idx, kz, operator=("grad", "abs"))
    with pytest.raises(ValueError, match="unsupported linked FFT operator"):
        _linked_fft_apply((f, f), idx, kz, operator=("grad", "curl"))


@pytest.mark.parametrize("beta", [0.0, 0.05])
def test_adiabatic_ion_zonal_field_solve_has_no_flux_surface_average(beta) -> None:
    """Boltzmann ions (kinetic electrons only) take GX ``qneutAdiab`` at ky = 0.

    Only adiabatic electrons respond to ``phi - <phi>`` (GX ``iphi00 = 2``);
    adiabatic ions respond to the full ``phi``, so the zonal solve is the local
    ``nbar / (tau + qneut)`` and cannot depend on the field-line Jacobian.
    """

    cache, params, G, _charge, density, temp, mass, _tz, vth = _build_case(
        beta=beta, fapar=0.0
    )
    charge = tz = jnp.asarray([-1.0], dtype=jnp.float32)
    G = G.at[0, :, 0, 0, 1, :].set(jnp.asarray(0.2 + 0.05j * cache.bmag, G.dtype))
    w_bpar = jnp.asarray(1.0 if beta else 0.0, dtype=jnp.float32)

    def phi(jacobian):
        return _solve_fields_impl(
            G,
            replace(cache, jacobian=jacobian),
            params,
            charge=charge,
            density=density,
            temp=temp,
            mass=mass,
            tz=tz,
            vth=vth,
            fapar=jnp.asarray(0.0, dtype=jnp.float32),
            w_bpar=w_bpar,
        ).phi

    varied = jnp.linspace(1.0, 3.0, cache.jacobian.size, dtype=cache.jacobian.dtype)
    base = phi(cache.jacobian)
    assert jnp.abs(base[0, 1, :]).max() > 0.0
    np.testing.assert_array_equal(np.asarray(phi(varied)), np.asarray(base))
    if beta == 0.0:
        jl = cache.Jl[0, :, 0, 1, :]
        nbar = -jnp.sum(jl * G[0, :, 0, 0, 1, :], axis=0)
        qneut = 1.0 - jnp.sum(jl * jl, axis=0)
        np.testing.assert_allclose(
            np.asarray(base[0, 1, :]),
            np.asarray(nbar / (params.tau_e + qneut)),
            rtol=1.0e-6,
        )


# #328: real-FFT parallel derivatives against analytic oracles.
@pytest.mark.parametrize("nz", [8, 9])
@pytest.mark.parametrize("nx", [1, 3])
@pytest.mark.parametrize("half", [False, True])
@pytest.mark.parametrize("dtype", [jnp.complex64, jnp.complex128])
def test_parallel_gradient_preserves_real_zonal_field(nz, nx, half, dtype):
    if dtype == jnp.complex128 and not jax.config.x64_enabled:
        pytest.skip("complex128 requires x64")
    ny = 8
    rows = ny // 2 + 1 if half else ny
    z = np.arange(nz) * (2 * np.pi / nz)
    nyq = (-1.0) ** np.arange(nz) * (nz % 2 == 0)
    f = np.zeros((rows, nx, nz), complex)
    expected = np.zeros_like(f)
    f[0, 0], expected[0, 0] = np.sin(z) + 0.7 * nyq, np.cos(z)
    if nx > 1:
        f[0, 1] = (1 + 2j) * np.cos(z) + (0.2 + 0.4j) * nyq
        expected[0, 1] = -(1 + 2j) * np.sin(z)
        f[0, -1], expected[0, -1] = f[0, 1].conj(), expected[0, 1].conj()
    kz = 2 * jnp.pi * jnp.fft.fftfreq(nz, d=2 * np.pi / nz)
    fj = jnp.asarray(f, dtype)
    linked = grad_z_linked_fft(
        fj,
        dz=2 * np.pi / nz,
        linked_indices=(jnp.asarray(np.arange(nx)[:, None] * rows, jnp.int32),),
        linked_kz=(kz,),
        ny_full=ny,
    )
    tol = 3e-6 if dtype == jnp.complex64 else 2e-13
    for observed in (linked, grad_z_periodic(fj, kz=kz, ny_full=ny)):
        np.testing.assert_allclose(observed, expected, atol=tol, rtol=tol)
    # Complex linearity, and the generic/selected-ky signed Nyquist is kept.
    np.testing.assert_allclose(
        grad_z_periodic(1j * fj, kz=kz, ny_full=ny), 1j * linked, atol=tol, rtol=tol
    )
    if nz % 2 == 0:
        wave = jnp.asarray(nyq, dtype)[None, None, :]
        for kw in ({}, {"ny_full": 24}):
            np.testing.assert_allclose(
                grad_z_periodic(wave, kz=kz, **kw), -(nz // 2) * 1j * wave, atol=tol
            )


def test_single_radial_mode_self_conjugate_row_is_real():
    value = jnp.ones((8, 1, 2), dtype=jnp.complex64) * (1 + 2j)
    result = symmetrize_self_conjugate_rows(value, ny_full=8)
    np.testing.assert_array_equal(result[0].imag, 0)
    np.testing.assert_array_equal(result[4].imag, 0)
    np.testing.assert_array_equal(result[1], value[1])


def _hermitian_multimode_case(nl, *, nu=0.0, seed=0):
    """Variable-B, sheared, three-field multimode case with a real state."""

    dtype = np.float64 if jax.config.x64_enabled else np.float32
    values, jax_values, rho, nt = _electromagnetic_species(dtype)
    fprim, tprim = np.array([0.8, 1.1]), np.array([2.4, 1.7])
    params = LinearParams(
        beta=0.04,
        fapar=1.0,
        tau_e=0.0,
        charge_sign=jax_values["charge"],
        **{k: jax_values[k] for k in ("density", "temp", "mass", "tz", "vth")},
        rho=jnp.asarray(rho, dtype),
        rho_star=0.8,
        kpar_scale=1.0 / (1.4 * 2.77778),
        fprim=jnp.asarray(fprim, dtype),
        tprim=jnp.asarray(tprim, dtype),
        nu=jnp.asarray([nu, 6.0 * nu], dtype),
        D_hyper=0.1 * nu,
    )
    geom = SAlphaGeometry(
        q=1.4,
        s_hat=0.8,
        epsilon=0.18,
        R0=2.77778,
        kperp2_bmag=True,
        bessel_bmag_power=1.0,
    )
    grid = build_spectral_grid(
        GridConfig(Nx=4, Ny=6, Nz=32, Lx=6.0, Ly=6.0, boundary="periodic")
    )
    cache = build_linear_cache(grid, geom, params, Nl=nl, Nm=4)
    rng = np.random.default_rng(seed)
    shape = (2, nl, 4, grid.ky.size, grid.kx.size, grid.z.size)
    G = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    G *= 0.01 * np.cos(np.asarray(grid.z))[None, None, None, None, None, :] ** 2
    # A real field: project through physical space, then drop the mean and
    # the self-conjugate Nyquist rows, which the energy measure treats apart.
    G = np.fft.fft2(np.fft.ifft2(G, axes=(3, 4)).real, axes=(3, 4))
    G[:, :, :, 0, 0] = G[:, :, :, grid.ky.size // 2] = G[:, :, :, :, 2] = 0.0
    ctype = jnp.complex128 if dtype == np.float64 else jnp.complex64
    vol, flux_fac = fieldline_quadrature_weights(geom, grid)
    return dict(
        G=jnp.asarray(G, ctype),
        grid=grid,
        cache=cache,
        params=params,
        values=values,
        jax_values=jax_values,
        nt=nt,
        vol=vol,
        flux_fac=flux_fac,
        fprim=fprim,
        tprim=tprim,
        dtype=dtype,
    )


def _free_energy(G, case):
    """Physical three-field free energy: entropy - Boltzmann + |dB|^2 / beta."""

    cache, params, jv, vol = case["cache"], case["params"], case["jax_values"], case["vol"]
    fields = _solve_fields_impl(G, cache, params, fapar=1.0, w_bpar=1.0, **jv)
    H = build_H(
        G, cache.Jl, fields.phi, jv["tz"], fields.apar, jv["vth"], fields.bpar, cache.JlB
    )
    entropy = 0.5 * jnp.sum(vol * case["nt"][:, None, None, None, None, None] * jnp.abs(H) ** 2)
    boltzmann = 0.5 * jnp.sum(vol * jnp.abs(fields.phi) ** 2) * jnp.sum(
        jv["density"] * jv["charge"] ** 2 / jv["temp"]
    )
    B2 = cache.bmag**2
    magnetic = jnp.sum(
        vol * B2 * (cache.kperp2 * jnp.abs(fields.apar) ** 2 + jnp.abs(fields.bpar) ** 2)
    )
    return entropy - boltzmann + magnetic / params.beta, H


def _drive_from_fluxes(case, fields, *, bpar):
    """rho_* sum_s [fprim T Gamma + tprim (Q - 3/2 T Gamma)] from flux diagnostics."""

    from gkx.operators.moments import (
        _heat_flux_channel_contrib_species,
        _particle_flux_channel_contrib_species,
    )

    args = (
        case["G"],
        fields.phi,
        fields.apar,
        fields.bpar if bpar else jnp.zeros_like(fields.phi),
        case["cache"],
        case["grid"],
        case["params"],
        case["flux_fac"],
    )
    kw = dict(use_dealias=False, flux_scale=1.0)
    Q = sum(jnp.sum(c, axis=(1, 2, 3)) for c in _heat_flux_channel_contrib_species(*args, **kw))
    gam = sum(jnp.sum(c, axis=(1, 2, 3)) for c in _particle_flux_channel_contrib_species(*args, **kw))
    T = case["values"]["temp"]
    fp, tp = case["fprim"], case["tprim"]
    return case["params"].rho_star * float(
        np.sum(fp * T * gam + tp * (np.asarray(Q) - 1.5 * T * np.asarray(gam)))
    )


@pytest.mark.parametrize("bpar", [0.0, 1.0])
def test_three_field_free_energy_budget_closes_multimode(bpar):
    """dW/dt = drive - dissipation for (phi, Apar, Bpar) at variable B.

    W is assembled from physical channels; its time derivative comes from
    forward-mode AD along the assembled RHS.  The drive comes independently
    from the particle and heat flux diagnostics.  Mirror/streaming and drifts
    are conservative; collisions and hyperdiffusion only dissipate.
    """

    case = _hermitian_multimode_case(3, nu=0.05)
    G, cache, params, dtype = case["G"], case["cache"], case["params"], case["dtype"]
    tol = 1e-12 if dtype == np.float64 else 2e-5
    terms = TermConfig(hyperdiffusion=1.0, end_damping=0.0, bpar=bpar)
    total, fields, contrib = assemble_rhs_terms_cached(
        G, cache, params, terms=terms, use_custom_vjp=False
    )
    W, H = _free_energy(G, case)
    if bpar:
        # The physical energy is the GX source quadratic 1/2 <G, nT H>.
        source = 0.5 * jnp.real(
            jnp.sum(case["vol"] * case["nt"][:, None, None, None, None, None] * jnp.conj(G) * H)
        )
        np.testing.assert_allclose(float(W), float(source), rtol=tol)
        _, dW = jax.jvp(lambda g: _free_energy(g, case)[0], (G,), (total,))
    weight = case["vol"] * case["nt"][:, None, None, None, None, None] * jnp.conj(H)
    rates = {k: float(jnp.real(jnp.sum(weight * v))) for k, v in contrib.items()}
    scale = float(jnp.sum(jnp.abs(weight * total)))
    if bpar:
        np.testing.assert_allclose(float(dW), sum(rates.values()), rtol=tol, atol=tol * scale)
    for name in ("curvature", "gradb"):
        assert abs(rates[name]) < tol * scale
    assert abs(rates["streaming"] + rates["mirror"]) < 1e-8 * scale
    assert abs(rates["streaming"]) > 1e3 * abs(rates["streaming"] + rates["mirror"])
    for name in ("collisions", "hypercollisions", "hyperdiffusion"):
        assert rates[name] < 0.0
    drive = _drive_from_fluxes(case, fields, bpar=bpar)
    # The phi-Bpar cross pairing of the diamagnetic drive uses the analytic
    # upper Laguerre neighbour (#325), so with Bpar it closes to truncation.
    np.testing.assert_allclose(
        rates["diamagnetic"], drive, rtol=1e-6 if bpar else 50 * tol
    )


def test_bpar_diamagnetic_budget_converges_with_laguerre_truncation():
    errors = []
    for nl in (3, 6, 10):
        case = _hermitian_multimode_case(nl)
        _, fields, contrib = assemble_rhs_terms_cached(
            case["G"], case["cache"], case["params"], use_custom_vjp=False
        )
        _, H = _free_energy(case["G"], case)
        rate = jnp.sum(
            case["vol"] * case["nt"][:, None, None, None, None, None]
            * jnp.conj(H) * contrib["diamagnetic"]
        )
        drive = _drive_from_fluxes(case, fields, bpar=True)
        errors.append(abs(float(jnp.real(rate)) - drive) / abs(drive))
    floor = 1e-13 if jax.config.x64_enabled else 1e-5
    assert errors[0] > 10 * errors[1] > 10 * max(errors[2], floor)
