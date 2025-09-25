import jax.numpy as jnp
from spectraxgk.dg import upwind_D
from spectraxgk.poisson import build_P


def test_periodic_ops_shapes():
    Nx, L = 16, 1.0
    P = build_P(Nx, L, "periodic")
    D = upwind_D(Nx, L, "periodic")
    assert P.shape == (Nx, Nx)
    assert D.shape == (Nx, Nx)
    assert jnp.allclose(jnp.sum(P, axis=1)[0], 0.0, atol=1e-10)  # neutrality at k=0
