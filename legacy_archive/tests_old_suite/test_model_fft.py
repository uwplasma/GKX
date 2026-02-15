# tests/test_model_fft.py
import numpy as np
import jax.numpy as jnp

from spectraxgk._model import fft_shifted, ifft_shifted


def test_fft_ifft_are_inverses_complex():
    Ny, Nx, Nz = 7, 9, 5
    rng = np.random.default_rng(0)
    re = rng.standard_normal((Ny, Nx, Nz))
    im = rng.standard_normal((Ny, Nx, Nz))
    A = jnp.array(re + 1j * im, dtype=jnp.complex128)

    Ak = fft_shifted(A)
    A2 = ifft_shifted(Ak)

    assert jnp.allclose(A2, A, rtol=1e-12, atol=1e-12)
