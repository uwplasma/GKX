"""Minimal linear run using the config-driven runner."""

import numpy as np
import jax.numpy as jnp

from gkx import (
    CycloneBaseCase,
    GridConfig,
    LinearParams,
    integrate_linear_from_config,
)
from gkx.geometry import SAlphaGeometry
from gkx.core.grid import build_spectral_grid


def main() -> None:
    grid_cfg = GridConfig(Nx=1, Ny=12, Nz=32, Lx=6.28, Ly=6.28)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()

    G0 = jnp.zeros((2, 2, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)
    G0 = G0.at[0, 0, 0, 0, :].set(1.0e-3 + 0.0j)

    _, phi_t = integrate_linear_from_config(G0, grid, geom, params, cfg.time)
    print("phi_t shape:", np.asarray(phi_t).shape)


if __name__ == "__main__":
    main()
