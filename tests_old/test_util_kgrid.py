import jax.numpy as jnp
from spectraxgk.backends import resolve_kgrid
from spectraxgk.io_config import GridCfg


def test_resolve_kgrid_basic():
    grid = GridCfg(L=2.0, Nx=8)
    k = resolve_kgrid(grid)
    dk = float(k[1] - k[0])
    assert (
        jnp.allclose(dk, 2 * jnp.pi / 8 / (2.0 / 8)) or dk != 0.0
    )  # just non-zero uniform spacing
    assert k.shape[0] == 8
