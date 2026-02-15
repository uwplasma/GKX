# tests/test_hl_basis.py
import numpy as np
import jax.numpy as jnp

from spectraxgk._hl_basis import (
    twothirds_mask,
    kgrid_fftshifted,
    laguerre_L_all,
    J_l_all,
    alpha_tensor,
    conjugate_index_fftshifted,
)


def test_conjugate_index_fftshifted_odd_even():
    # odd N: i' = (N-1)-i
    N = 5
    for i in range(N):
        assert conjugate_index_fftshifted(i, N) == (N - 1) - i

    # even N: i' = (N - i) mod N
    N = 6
    for i in range(N):
        assert conjugate_index_fftshifted(i, N) == ((N - i) % N)


def test_twothirds_mask_shape_and_center_kept():
    Ny, Nx, Nz = 9, 9, 7
    m = twothirds_mask(Ny, Nx, Nz)
    assert m.shape == (Ny, Nx, Nz)

    # center (k=0) should be kept
    assert bool(m[Ny // 2, Nx // 2, Nz // 2]) is True


def test_kgrid_fftshifted_shapes_and_units():
    Lx, Ly, Lz = 2 * np.pi, 4 * np.pi, 6 * np.pi
    Nx, Ny, Nz = 8, 6, 10
    kx, ky, kz = kgrid_fftshifted(Lx, Ly, Lz, Nx, Ny, Nz)
    assert kx.shape == (Ny, Nx, Nz)
    assert ky.shape == (Ny, Nx, Nz)
    assert kz.shape == (Ny, Nx, Nz)

    # k=0 at center in fftshift ordering
    assert float(kx[Ny // 2, Nx // 2, Nz // 2]) == 0.0
    assert float(ky[Ny // 2, Nx // 2, Nz // 2]) == 0.0
    assert float(kz[Ny // 2, Nx // 2, Nz // 2]) == 0.0


def test_laguerre_L_all_branches_Nl_1_2_and_recurrence():
    b = jnp.linspace(0.0, 3.0, 11)

    L1 = laguerre_L_all(b, 1)
    assert L1.shape == (1, b.size)
    assert jnp.allclose(L1[0], 1.0)

    L2 = laguerre_L_all(b, 2)
    assert L2.shape == (2, b.size)
    assert jnp.allclose(L2[0], 1.0)
    assert jnp.allclose(L2[1], 1.0 - b)

    # recurrence spot-check for Nl=5
    Nl = 5
    L = laguerre_L_all(b, Nl)
    # check recurrence for l=1 -> L2
    l = 1.0
    Lp = ((2.0 * l + 1.0 - b) * L[1] - l * L[0]) / (l + 1.0)
    assert jnp.allclose(L[2], Lp, rtol=1e-12, atol=1e-12)


def test_J_l_all_definition():
    b = jnp.array([0.0, 0.1, 2.0])
    Nl = 4
    L = laguerre_L_all(b, Nl)
    J = J_l_all(b, Nl)
    assert jnp.allclose(J, jnp.exp(-0.5 * b)[None, ...] * L, rtol=1e-14, atol=1e-14)


def test_alpha_tensor_basic_properties():
    Nl = 6
    A = alpha_tensor(Nl)
    assert A.shape == (Nl, Nl, Nl)
    assert A.dtype == jnp.float64

    # non-negativity (constructed by exp of logs)
    assert float(jnp.min(A)) >= 0.0

    # triangle inequality support: if n < |k-l| or n > k+l -> coefficient should be 0
    # spot-check a few
    assert float(A[5, 0, 0]) == 0.0  # n=0 < |5-0|=5
    assert float(A[3, 3, 0]) >= 0.0  # allowed since |3-3|=0
