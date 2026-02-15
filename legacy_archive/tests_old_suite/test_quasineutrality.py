# tests/test_quasineutrality.py
import jax.numpy as jnp

from spectraxgk._model import solve_phi_from_quasineutrality_boltzmann_e


def test_phi_k0_is_zero(tiny_params, tiny_state):
    phi = solve_phi_from_quasineutrality_boltzmann_e(tiny_state, tiny_params)
    Ny, Nx, Nz = phi.shape
    assert phi[Ny // 2, Nx // 2, Nz // 2] == 0.0 + 0.0j


def test_phi_is_dealiased(tiny_params, tiny_state):
    phi = solve_phi_from_quasineutrality_boltzmann_e(tiny_state, tiny_params)
    mask = tiny_params["mask23"]
    # phi should be identically zero where mask is false
    assert jnp.allclose(phi * (1 - mask), 0.0 + 0.0j)
