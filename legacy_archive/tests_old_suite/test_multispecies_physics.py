import pytest
import jax
import jax.numpy as jnp

from spectraxgk._initialization_multispecies import initialize_simulation_parameters_multispecies
from spectraxgk._model_multispecies import (
    rhs_gk_multispecies,
    solve_phi_quasineutrality_multispecies,
    collision_lenard_bernstein_conserving_multispecies,
    enforce_conjugate_symmetry_fftshifted,
)


def _symmetry_error_fftshifted(Ak, params):
    iy, ix, iz = params["conj_y"], params["conj_x"], params["conj_z"]
    Aneg = jnp.take(Ak, iy, axis=-3)
    Aneg = jnp.take(Aneg, ix, axis=-2)
    Aneg = jnp.take(Aneg, iz, axis=-1)
    denom = jnp.linalg.norm(Ak.ravel()) + 1e-30
    return jnp.linalg.norm((Ak - jnp.conj(Aneg)).ravel()) / denom


@pytest.fixture(scope="module")
def params_linear():
    # Nonlinear off to isolate linear physics; small sizes.
    Nx, Ny, Nz = 5, 5, 9
    Nl, Nh = 2, 10
    species = [
        dict(name="ion", q=+1.0, T=1.0, n0=1.0, rho=1.0, vth=1.0, nu=0.0, Upar=0.0),
        dict(name="e",   q=-1.0, T=1.0, n0=1.0, rho=1.0, vth=1.0, nu=0.0, Upar=0.0),
    ]
    p = initialize_simulation_parameters_multispecies(
        dict(species=species, enable_nonlinear=False, enable_collisions=False, enforce_reality=True),
        Nx=Nx, Ny=Ny, Nz=Nz, Nl=Nl, Nh=Nh, timesteps=10, dt=1e-2
    )
    return p


def test_rhs_preserves_conjugate_symmetry(params_linear):
    p = params_linear
    key = jax.random.PRNGKey(0)
    G = jax.random.normal(key, (p["Ns"], p["Nl"], p["Nh"], p["Ny"], p["Nx"], p["Nz"])) \
        + 1j * jax.random.normal(key, (p["Ns"], p["Nl"], p["Nh"], p["Ny"], p["Nx"], p["Nz"]))
    G = G.astype(p["Gk_0"].dtype)

    # Symmetrize input and check output stays symmetric
    G = enforce_conjugate_symmetry_fftshifted(G, p)
    dG, phi = rhs_gk_multispecies(G, p, Nh=p["Nh"], Nl=p["Nl"])

    assert float(_symmetry_error_fftshifted(dG, p)) < 1e-6
    assert float(_symmetry_error_fftshifted(phi[None, None, None, ...], p)) < 1e-6  # embed to reuse helper logic


def test_streaming_is_energy_conserving_linear(params_linear):
    """
    Correct invariant here is W(G)=0.5||H(G,phi(G))||^2, not <G,dG>.
    We check directional derivative dW/dt ≈ 0 via finite difference along RHS.
    """
    p = params_linear
    key = jax.random.PRNGKey(1)

    G = jax.random.normal(key, (p["Ns"], p["Nl"], p["Nh"], p["Ny"], p["Nx"], p["Nz"])) \
        + 1j * jax.random.normal(key, (p["Ns"], p["Nl"], p["Nh"], p["Ny"], p["Nx"], p["Nz"]))
    G = G.astype(p["Gk_0"].dtype)

    # Project onto the same manifold the RHS uses: dealias + enforce reality.
    def project(Gx):
        mask = p.get("mask23_c", p["mask23"].astype(Gx.dtype))
        Gx = Gx * mask[None, None, None, ...]
        return enforce_conjugate_symmetry_fftshifted(Gx, p)

    G = project(G)
    dG, _phi = rhs_gk_multispecies(G, p, Nh=p["Nh"], Nl=p["Nl"])

    from spectraxgk._model_multispecies import cheap_diagnostics_multispecies

    eps_val = 1e-6 if (G.real.dtype == jnp.float64) else 1e-4
    eps = jnp.asarray(eps_val, dtype=G.real.dtype)


    Wplus  = cheap_diagnostics_multispecies(project(G + eps * dG), p)["W_free"]
    Wminus = cheap_diagnostics_multispecies(project(G - eps * dG), p)["W_free"]
    dW = (Wplus - Wminus) / (2*eps)

    assert float(jnp.abs(dW)) < 1e-3



def test_collisions_are_dissipative():
    """
    For random H, the conserving Lenard–Bernstein operator should satisfy
      -Re <H, C(H)> >= 0
    """
    Nx, Ny, Nz = 3, 3, 5
    Nl, Nh = 2, 8
    species = [
        dict(name="ion", q=+1.0, T=1.0, n0=1.0, rho=1.0, vth=1.0, nu=0.2, Upar=0.0),
        dict(name="e",   q=-1.0, T=1.0, n0=1.0, rho=1.0, vth=1.0, nu=0.2, Upar=0.0),
    ]
    p = initialize_simulation_parameters_multispecies(
        dict(species=species, enable_nonlinear=False, enforce_reality=False),
        Nx=Nx, Ny=Ny, Nz=Nz, Nl=Nl, Nh=Nh, timesteps=10, dt=1e-2
    )

    key = jax.random.PRNGKey(2)
    H = jax.random.normal(key, (p["Ns"], p["Nl"], p["Nh"], p["Ny"], p["Nx"], p["Nz"])) \
        + 1j * jax.random.normal(key, (p["Ns"], p["Nl"], p["Nh"], p["Ny"], p["Nx"], p["Nz"]))
    H = H.astype(p["Gk_0"].dtype)

    C = collision_lenard_bernstein_conserving_multispecies(H, p)
    diss = -jnp.real(jnp.vdot(H, C))
    assert float(diss) >= -1e-6
