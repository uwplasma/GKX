"""Tests for modular field solves and custom-VJP behavior."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp

from gkx.config import CycloneBaseCase, GridConfig
from gkx.geometry import SAlphaGeometry
from gkx.core.grid import build_spectral_grid
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.moments import quasineutrality_phi
from gkx.operators.linear.params import LinearParams
from gkx.terms.fields import _solve_fields_impl, solve_fields


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
    z = jnp.asarray(grid.z)
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
