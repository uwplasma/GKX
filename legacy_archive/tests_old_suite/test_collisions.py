# tests/test_collisions.py
import jax.numpy as jnp

from spectraxgk._model import collision_lenard_bernstein_conserving, solve_phi_from_quasineutrality_boltzmann_e


def test_collisions_nu0_fast_path_returns_zero(tiny_params, tiny_state):
    p = dict(tiny_params)
    p["nu"] = 0.0

    phi = solve_phi_from_quasineutrality_boltzmann_e(tiny_state, p)
    Hk = tiny_state.at[:, 0, ...].add(p["Jl_grid"] * phi)

    C = collision_lenard_bernstein_conserving(Hk, p)
    assert jnp.allclose(C, 0.0 + 0.0j)


def test_collisions_produce_dissipation_nonnegative(tiny_params, tiny_state):
    p = dict(tiny_params)
    p["nu"] = 0.1

    phi = solve_phi_from_quasineutrality_boltzmann_e(tiny_state, p)
    Hk = tiny_state.at[:, 0, ...].add(p["Jl_grid"] * phi)

    C = collision_lenard_bernstein_conserving(Hk, p)
    # Dissipation proxy: -Re <H, C(H)> >= 0 for a dissipative operator
    D = -jnp.real(jnp.vdot(Hk, C))
    assert float(D) >= -1e-10  # allow tiny roundoff
