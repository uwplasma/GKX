import pytest
import jax
import jax.numpy as jnp

from spectraxgk._initialization_multispecies import initialize_simulation_parameters_multispecies
from spectraxgk._model_multispecies import (
    enforce_conjugate_symmetry_fftshifted,
    solve_phi_quasineutrality_multispecies,
    shift_m,
)


def _symmetry_error_fftshifted(Ak, params):
    """||A - conj(A(-k))|| / ||A|| in fftshift ordering."""
    iy, ix, iz = params["conj_y"], params["conj_x"], params["conj_z"]
    Aneg = jnp.take(Ak, iy, axis=-3)
    Aneg = jnp.take(Aneg, ix, axis=-2)
    Aneg = jnp.take(Aneg, iz, axis=-1)
    denom = jnp.linalg.norm(Ak.ravel()) + 1e-30
    return jnp.linalg.norm((Ak - jnp.conj(Aneg)).ravel()) / denom


@pytest.fixture(scope="module")
def small_params():
    # Small, fast grid; includes nonzero k_perp modes.
    Nx, Ny, Nz = 5, 5, 7
    Nl, Nh = 3, 6
    species = [
        dict(name="ion", q=+1.0, T=1.0, n0=1.0, rho=1.0, vth=1.0, nu=0.0, Upar=0.0),
        dict(name="e",   q=-1.0, T=1.0, n0=1.0, rho=1.0, vth=1.0, nu=0.0, Upar=0.0),
    ]
    p = initialize_simulation_parameters_multispecies(
        dict(species=species, t_max=1.0, enable_nonlinear=False),
        Nx=Nx, Ny=Ny, Nz=Nz, Nl=Nl, Nh=Nh, timesteps=10, dt=1e-2
    )
    return p


def test_shift_m_shapes_and_edges(small_params):
    p = small_params
    Ns, Nl, Nh, Ny, Nx, Nz = p["Ns"], p["Nl"], p["Nh"], p["Ny"], p["Nx"], p["Nz"]
    H = jnp.arange(Ns * Nl * Nh * Ny * Nx * Nz, dtype=jnp.float32).reshape((Ns, Nl, Nh, Ny, Nx, Nz))
    H = H.astype(jnp.complex64) + 0.0j

    Hp1 = shift_m(H, +1)
    Hm1 = shift_m(H, -1)

    assert Hp1.shape == H.shape
    assert Hm1.shape == H.shape

    # dm=+1: last m slice must be 0
    assert jnp.allclose(Hp1[:, :, -1, ...], 0.0)
    # dm=-1: first m slice must be 0
    assert jnp.allclose(Hm1[:, :, 0, ...], 0.0)


def test_enforce_conjugate_symmetry_idempotent(small_params):
    p = small_params
    key = jax.random.PRNGKey(0)
    G = jax.random.normal(key, (p["Ns"], p["Nl"], p["Nh"], p["Ny"], p["Nx"], p["Nz"])) \
        + 1j * jax.random.normal(key, (p["Ns"], p["Nl"], p["Nh"], p["Ny"], p["Nx"], p["Nz"]))
    G = G.astype(p["Gk_0"].dtype)

    Gsym = enforce_conjugate_symmetry_fftshifted(G, p)
    err1 = _symmetry_error_fftshifted(Gsym, p)
    assert float(err1) < 1e-6

    # applying again should not change it (up to tiny roundoff)
    Gsym2 = enforce_conjugate_symmetry_fftshifted(Gsym, p)
    assert jnp.allclose(Gsym2, Gsym, atol=1e-6, rtol=1e-6)


def test_quasineutrality_gauge_and_dealias(small_params):
    p = small_params
    G0 = p["Gk_0"]
    phi = solve_phi_quasineutrality_multispecies(G0, p)

    # gauge mode should be exactly 0
    Ny, Nx, Nz = p["Ny"], p["Nx"], p["Nz"]
    assert phi[Ny//2, Nx//2, Nz//2] == 0.0 + 0.0j

    # de-alias: phi must be zero wherever mask23 is False
    mask23 = p["mask23"]
    assert jnp.allclose(phi * (~mask23), 0.0 + 0.0j)


def test_quasineutrality_charge_cancellation(small_params):
    """
    If we set g_{m=0} so that Σ_s q_s n0_s Σ_l J_l g = 0, then phi should be ~0.
    Easiest: choose identical g for two species with equal/opposite q and equal n0 -> cancels.
    """
    p = small_params

    # Build a state with identical g for ion and electron; with q=±1 and n0 equal this cancels numerator.
    key = jax.random.PRNGKey(1)
    g = jax.random.normal(key, (p["Nl"], p["Ny"], p["Nx"], p["Nz"])) \
        + 1j * jax.random.normal(key, (p["Nl"], p["Ny"], p["Nx"], p["Nz"]))
    # g = g.astype(jnp.complex64)

    G = jnp.zeros((p["Ns"], p["Nl"], p["Nh"], p["Ny"], p["Nx"], p["Nz"]), dtype=jnp.complex64)
    G = G.at[0, :, 0, ...].set(g)  # ion
    G = G.at[1, :, 0, ...].set(g)  # electron (same g)

    phi = solve_phi_quasineutrality_multispecies(G, p)
    assert float(jnp.max(jnp.abs(phi))) < 1e-5
