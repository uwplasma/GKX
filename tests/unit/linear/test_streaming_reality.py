"""Real Fourier fields have a cosine, not a signed complex Nyquist wave."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gkx.core_ky_layout import reality_residual, symmetrize_self_conjugate_rows
from gkx.operators.linear.streaming import grad_z_linked_fft, grad_z_periodic


@pytest.mark.parametrize("nz", [8, 9])
@pytest.mark.parametrize("nx", [1, 3])
@pytest.mark.parametrize("half", [False, True])
@pytest.mark.parametrize("dtype", [jnp.complex64, jnp.complex128])
def test_parallel_gradient_preserves_real_zonal_field(nz, nx, half, dtype):
    if dtype == jnp.complex128 and not jax.config.x64_enabled:
        pytest.skip("complex128 requires x64")
    ny = 8
    rows = ny // 2 + 1 if half else ny
    z = np.arange(nz) * (2 * np.pi / nz)
    f = np.zeros((rows, nx, nz), complex)
    expected = np.zeros_like(f)
    f[0, 0] = np.sin(z)
    expected[0, 0] = np.cos(z)
    if nz % 2 == 0:
        f[0, 0] += 0.7 * (-1.0) ** np.arange(nz)
    if nx > 1:
        f[0, 1] = (1 + 2j) * np.cos(z)
        if nz % 2 == 0:
            f[0, 1] += (0.2 + 0.4j) * (-1.0) ** np.arange(nz)
        f[0, -1] = f[0, 1].conj()
        expected[0, 1] = -(1 + 2j) * np.sin(z)
        expected[0, -1] = expected[0, 1].conj()
    indices = jnp.asarray(np.arange(nx)[:, None] * rows, jnp.int32)
    kz = 2 * jnp.pi * jnp.fft.fftfreq(nz, d=2 * np.pi / nz)
    linked = grad_z_linked_fft(
        jnp.asarray(f, dtype),
        dz=2 * np.pi / nz,
        linked_indices=(indices,),
        linked_kz=(kz,),
        ny_full=ny,
    )
    periodic = grad_z_periodic(jnp.asarray(f, dtype), kz=kz, ny_full=ny)
    tol = 3e-6 if dtype == jnp.complex64 else 2e-13
    np.testing.assert_allclose(linked, expected, atol=tol, rtol=tol)
    np.testing.assert_allclose(periodic, expected, atol=tol, rtol=tol)
    if not half:
        assert float(reality_residual(linked)) < tol


def test_single_radial_mode_self_conjugate_row_is_real():
    value = jnp.ones((8, 1, 2), dtype=jnp.complex64) * (1 + 2j)
    result = symmetrize_self_conjugate_rows(value, ny_full=8)
    np.testing.assert_array_equal(result[0].imag, 0)
    np.testing.assert_array_equal(result[4].imag, 0)
    np.testing.assert_array_equal(result[1], value[1])


def test_selected_complex_mode_keeps_signed_nyquist_derivative():
    nz = 8
    wave = jnp.asarray((-1.0) ** np.arange(nz), jnp.complex64)[None, None, :]
    kz = 2 * jnp.pi * jnp.fft.fftfreq(nz, d=2 * np.pi / nz)
    expected = -4j * wave
    # One selected positive ky is not a complete real-FFT layout.
    np.testing.assert_allclose(grad_z_periodic(wave, kz=kz, ny_full=24), expected)
    np.testing.assert_allclose(grad_z_periodic(wave, kz=kz), expected)
