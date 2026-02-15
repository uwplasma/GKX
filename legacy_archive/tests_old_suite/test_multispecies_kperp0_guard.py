import jax.numpy as jnp
from spectraxgk._initialization_multispecies import initialize_simulation_parameters_multispecies
from spectraxgk._model_multispecies import solve_phi_quasineutrality_multispecies

def test_kperp0_mode_is_finite_and_zeroed():
    Nx, Ny, Nz = 1, 1, 9
    Nl, Nh = 2, 6

    species = [
        dict(name="ion", q=+1.0, T=1.0, n0=1.0, rho=1.0, vth=1.0, nu=0.0, Upar=0.0),
        dict(name="e",   q=-1.0, T=1.0, n0=1.0, rho=1.0, vth=1.0, nu=0.0, Upar=0.0),
    ]
    p = initialize_simulation_parameters_multispecies(
        dict(species=species, enable_nonlinear=False),
        Nx=Nx, Ny=Ny, Nz=Nz, Nl=Nl, Nh=Nh, timesteps=10, dt=1e-2
    )

    G = p["Gk_0"]
    phi = solve_phi_quasineutrality_multispecies(G, p)

    assert jnp.all(jnp.isfinite(jnp.real(phi)))
    assert jnp.all(jnp.isfinite(jnp.imag(phi)))

    # With Nx=Ny=1, k_perp = 0 everywhere; den=0 => phi should be exactly 0 by guard.
    assert jnp.allclose(phi, 0.0 + 0.0j)
