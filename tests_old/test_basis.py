import jax.numpy as jnp
from spectraxgk_old.basis import hermite_coupling_factors, lb_eigenvalues


def test_hermite_coupling_shapes_and_values():
    Nn = 5
    s, sp1 = hermite_coupling_factors(Nn)
    assert s.shape == (Nn,)
    assert sp1.shape == (Nn,)
    # sqrt(0)=0, sqrt(1)=1
    assert s[0] == 0.0
    assert jnp.isclose(s[1], 1.0)
    assert jnp.isclose(sp1[0], 1.0)


def test_lb_eigenvalues_simple_linear_form():
    Nn, Nm = 4, 3
    lam = lb_eigenvalues(Nn, Nm, alpha=1.0, beta=2.0)
    # lambda_{n,m} = n + 2 m
    assert lam.shape == (Nn, Nm)
    assert jnp.isclose(lam[0, 0], 0.0)
    assert jnp.isclose(lam[1, 0], 1.0)
    assert jnp.isclose(lam[0, 1], 2.0)
