# tests/test_conjugate_symmetry.py
import jax.numpy as jnp

from spectraxgk._model import enforce_conjugate_symmetry_fftshifted, ifft_shifted


def _conj_map(G, params):
    iy, ix, iz = params["conj_y"], params["conj_x"], params["conj_z"]
    Gneg = jnp.take(G, iy, axis=-3)
    Gneg = jnp.take(Gneg, ix, axis=-2)
    Gneg = jnp.take(Gneg, iz, axis=-1)
    return jnp.conj(Gneg)


def test_enforce_conjugate_symmetry_property(tiny_params, tiny_state):
    G = enforce_conjugate_symmetry_fftshifted(tiny_state, tiny_params)
    G_conj = _conj_map(G, tiny_params)
    assert jnp.allclose(G, G_conj, rtol=1e-12, atol=1e-12)


def test_enforce_conjugate_symmetry_makes_ifft_more_real(tiny_params, tiny_state):
    A = tiny_state[0, 0]  # (Ny,Nx,Nz)
    A2 = enforce_conjugate_symmetry_fftshifted(A[None, None, ...], tiny_params)[0, 0]
    a_phys = ifft_shifted(A2)
    assert jnp.max(jnp.abs(jnp.imag(a_phys))) < 1e-10
