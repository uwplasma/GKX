# tests/test_rhs_toggles_and_shapes.py
import jax.numpy as jnp

from spectraxgk._model import rhs_laguerre_hermite_gk


def test_rhs_shapes(tiny_params, tiny_state):
    Nl, Nh, Ny, Nx, Nz = tiny_state.shape
    dGk, phi = rhs_laguerre_hermite_gk(tiny_state, tiny_params, Nh=Nh, Nl=Nl)
    assert dGk.shape == tiny_state.shape
    assert phi.shape == (Ny, Nx, Nz)


def test_rhs_toggle_branches(tiny_params, tiny_state):
    Nl, Nh, *_ = tiny_state.shape

    # all off -> derivative should be 0 (since only NL + stream + coll contribute)
    p = dict(tiny_params)
    p["enable_streaming"] = False
    p["enable_nonlinear"] = False
    p["enable_collisions"] = False

    dGk, _ = rhs_laguerre_hermite_gk(tiny_state, p, Nh=Nh, Nl=Nl)
    assert jnp.allclose(dGk, 0.0 + 0.0j)

    # enforce_reality False branch
    p2 = dict(p)
    p2["enable_streaming"] = True
    p2["enforce_reality"] = False
    dGk2, _ = rhs_laguerre_hermite_gk(tiny_state, p2, Nh=Nh, Nl=Nl)
    assert dGk2.shape == tiny_state.shape
